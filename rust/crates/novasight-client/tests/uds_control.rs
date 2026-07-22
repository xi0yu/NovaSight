use std::path::PathBuf;

use novasight_api::{build_control_router, build_control_router_with_services};
use novasight_client::{ClientError, ControlClient};
use novasight_core::RuntimeEpoch;
use novasight_runtime::{
    ConfigService, MotionProfileHub, MotionProfileStatus, PipelineState, PreviewHub,
    RuntimeDependencies, RuntimeSupervisor,
};
use novasight_store::config::YamlConfigRepository;
use novasight_store::motion_profile::MotionProfileRepository;

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
async fn typed_client_uses_the_same_motion_profile_authority_as_web() {
    let socket = SocketPath::new();
    let root = socket.0.with_extension("motion");
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind Unix control socket");
    let hub = MotionProfileHub::default();
    let repository = MotionProfileRepository::open(&root).unwrap();
    let (supervisor, runtime) = RuntimeSupervisor::spawn(
        RuntimeDependencies::recording().with_motion_profiles(hub, repository),
    );
    let server_runtime = runtime.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, build_control_router(server_runtime))
            .await
            .expect("serve Unix control socket")
    });
    let client = ControlClient::new(&socket.0);

    assert_eq!(
        client.motion_profile_status().await.unwrap(),
        MotionProfileStatus {
            enabled: false,
            trajectory_source: "static".to_owned(),
            active_profile: String::new(),
            profile_name: String::new(),
            sample_count: 0,
            profile_version: 0,
            spatial_curve_available: false,
            effective_runtime_parameters: Default::default(),
            source: Default::default(),
            revision: 0,
        }
    );
    assert_eq!(
        client
            .activate_builtin_motion()
            .await
            .unwrap()
            .active_profile,
        "builtin"
    );
    assert!(!client.disable_motion_profile().await.unwrap().enabled);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    server.abort();
    std::fs::remove_dir_all(root).unwrap();
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
