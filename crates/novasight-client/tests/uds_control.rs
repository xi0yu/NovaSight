use std::path::PathBuf;

use novasight_api::{
    build_control_router, build_control_router_with_control_plane,
    build_control_router_with_services, with_trusted_local_control,
};
use novasight_client::{ClientError, ControlClient};
use novasight_core::RuntimeEpoch;
use novasight_runtime::{
    ConfigService, PipelineState, PreviewHub, RuntimeDependencies, RuntimeSupervisor,
};
use novasight_store::config::YamlConfigRepository;
use novasight_store::license::{FileLicenseRepository, LicensePolicy};

struct SocketPath(PathBuf);

impl SocketPath {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "novasight-client-{}-{}.sock",
            std::process::id(),
            unique_id()
        )))
    }
}

impl Drop for SocketPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn unique_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[tokio::test]
async fn typed_client_drives_the_same_runtime_over_a_unix_socket() {
    let socket = SocketPath::new();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind Unix control socket");
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let server_runtime = runtime.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, build_control_router(server_runtime))
            .await
            .expect("serve Unix control socket")
    });
    let client = ControlClient::new(&socket.0);

    assert_eq!(
        client.status().await.expect("status").pipeline.state,
        PipelineState::Stopped
    );
    assert_eq!(
        client.start().await.expect("start").pipeline.state,
        PipelineState::Running
    );
    assert_eq!(
        client.restart().await.expect("restart").pipeline.state,
        PipelineState::Running
    );
    assert_eq!(
        client
            .emergency_stop()
            .await
            .expect("emergency stop")
            .pipeline
            .state,
        PipelineState::Stopped
    );

    runtime
        .shutdown_daemon()
        .await
        .expect("shutdown supervisor");
    supervisor.join().await.expect("join supervisor");
    let rejected = client.start().await.expect_err("closed supervisor rejects");
    assert!(matches!(
        rejected,
        ClientError::Daemon { ref code, .. } if code == "supervisor_closed"
    ));
    server.abort();
}

#[tokio::test]
async fn typed_client_requests_daemon_shutdown_over_trusted_local_control() {
    let socket = SocketPath::new();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind Unix control socket");
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let server_runtime = runtime.clone();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            with_trusted_local_control(build_control_router(server_runtime)),
        )
        .await
        .expect("serve trusted Unix control socket")
    });
    let client = ControlClient::new(&socket.0);

    let response = client.shutdown_daemon().await.expect("shutdown daemon");
    assert_eq!(response["shutdown"], true);

    supervisor.join().await.expect("join supervisor");
    server.abort();
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn typed_client_connects_over_a_linux_abstract_socket() {
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::{SocketAddr, UnixListener as StdUnixListener};

    let name = format!("novasight-client-{}-{}", std::process::id(), unique_id());
    let address = SocketAddr::from_abstract_name(name.as_bytes()).expect("abstract address");
    let std_listener = StdUnixListener::bind_addr(&address).expect("bind abstract socket");
    std_listener
        .set_nonblocking(true)
        .expect("make abstract listener nonblocking");
    let listener = tokio::net::UnixListener::from_std(std_listener).expect("Tokio listener");
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let server_runtime = runtime.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, build_control_router(server_runtime))
            .await
            .expect("serve abstract control socket")
    });
    let configured = PathBuf::from(format!("@{name}"));
    let client = ControlClient::new(&configured);

    assert_eq!(
        client.status().await.expect("status").pipeline.state,
        PipelineState::Stopped
    );
    assert!(!configured.exists());

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn typed_client_reads_and_updates_persisted_config_over_the_same_socket() {
    let socket = SocketPath::new();
    let config_path = socket.0.with_extension("yaml");
    std::fs::write(
        &config_path,
        "revision: 9\nserver:\n  port: 5174\nfuture:\n  retained: true\n",
    )
    .unwrap();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind Unix control socket");
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let service = ConfigService::new(
        &config_path,
        YamlConfigRepository::load(&config_path).unwrap(),
    );
    let app = build_control_router_with_services(runtime.clone(), Some(service), None);
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve Unix control socket")
    });
    let client = ControlClient::new(&socket.0);

    assert_eq!(client.config().await.unwrap().revision, 9);
    let update = client
        .update_config_field("server", "port", serde_json::json!(6001), Some(9))
        .await
        .unwrap();
    assert_eq!(update.config.revision, 10);
    assert_eq!(update.config.server.port, 6001);
    assert!(update.restart_required);
    assert!(!update.applied);
    assert_eq!(client.config().await.unwrap().revision, 10);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    server.abort();
    std::fs::remove_file(config_path).unwrap();
}

#[tokio::test]
async fn typed_client_controls_the_daemon_owned_preview_gate() {
    let socket = SocketPath::new();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind Unix control socket");
    let preview = PreviewHub::new(true);
    preview.begin_epoch(RuntimeEpoch(3));
    let (supervisor, runtime) =
        RuntimeSupervisor::spawn(RuntimeDependencies::recording().with_preview(preview));
    let server_runtime = runtime.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, build_control_router(server_runtime))
            .await
            .expect("serve Unix control socket")
    });
    let client = ControlClient::new(&socket.0);

    assert!(!client.preview_status().await.unwrap().active);
    assert!(client.set_preview_active(true).await.unwrap().active);
    assert!(!client.set_preview_active(false).await.unwrap().active);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn missing_socket_is_a_typed_connection_error() {
    let socket = SocketPath::new();
    let error = ControlClient::new(&socket.0)
        .status()
        .await
        .expect_err("missing daemon fails");

    assert!(matches!(error, ClientError::Connect { .. }));
    assert_eq!(error.code(), "daemon_connect_failed");
}

#[tokio::test]
async fn an_unresponsive_daemon_is_bounded_by_the_client_timeout() {
    let socket = SocketPath::new();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind Unix control socket");
    let server = tokio::spawn(async move {
        let (_stream, _) = listener.accept().await.expect("accept client");
        std::future::pending::<()>().await;
    });

    let error = ControlClient::new(&socket.0)
        .with_timeout(std::time::Duration::from_millis(50))
        .status()
        .await
        .expect_err("unresponsive daemon must time out");

    assert!(matches!(error, ClientError::Timeout(_)));
    assert_eq!(error.code(), "daemon_request_timed_out");
    server.abort();
}

#[tokio::test]
async fn license_middleware_rejection_remains_a_typed_daemon_error() {
    let socket = SocketPath::new();
    let license_path = socket.0.with_extension("license.json");
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind Unix control socket");
    let license = FileLicenseRepository::new(&license_path, LicensePolicy::new(true, None));
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = with_trusted_local_control(build_control_router_with_control_plane(
        runtime.clone(),
        None,
        license,
        None,
        false,
        None,
    ));
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve Unix control socket")
    });

    let error = ControlClient::new(&socket.0)
        .status()
        .await
        .expect_err("unlicensed status is rejected");

    assert!(matches!(
        error,
        ClientError::Daemon {
            status: hyper::StatusCode::UNAUTHORIZED,
            ref code,
            ref message,
        } if code == "LICENSE_REQUIRED" && message == "a valid license is required"
    ));
    assert_eq!(error.code(), "daemon_command_rejected");

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn detail_only_compatibility_error_keeps_http_status_and_reason() {
    let socket = SocketPath::new();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind Unix control socket");
    let app = axum::Router::new().route(
        "/api/v1/status",
        axum::routing::get(|| async {
            (
                axum::http::StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({"detail": "license required"})),
            )
        }),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve Unix control socket")
    });

    let error = ControlClient::new(&socket.0)
        .status()
        .await
        .expect_err("compatibility rejection is typed");

    assert!(matches!(
        error,
        ClientError::Daemon {
            status: hyper::StatusCode::UNAUTHORIZED,
            ref code,
            ref message,
        } if code == "HTTP_401" && message == "license required"
    ));
    server.abort();
}
