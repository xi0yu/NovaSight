use std::path::PathBuf;

use novasight_api::build_control_router;
use novasight_client::{ClientError, ControlClient};
use novasight_runtime::{PipelineState, RuntimeSupervisor};

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
