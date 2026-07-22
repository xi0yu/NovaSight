use std::sync::Arc;

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use novasight_api::{build_control_router, build_control_router_with_shutdown};
use novasight_core::{Clock, MonotonicNanos, PointerDevice, RecordingPointerDevice};
use novasight_pipeline::PipelineConfig;
use novasight_runtime::{PipelineState, RuntimeDependencies, RuntimeHandle, RuntimeSupervisor};
use serde_json::Value;
use tower::ServiceExt;

async fn request(runtime: &RuntimeHandle, method: &str, path: &str) -> (StatusCode, Value) {
    let app = build_control_router(runtime.clone());
    let request = Request::builder()
        .method(method)
        .uri(path)
        .body(Body::empty())
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    let body = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).expect("JSON body")
    };
    (status, body)
}

async fn shutdown(supervisor: RuntimeSupervisor, runtime: &RuntimeHandle) {
    runtime.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("join supervisor");
}

#[tokio::test]
async fn status_is_the_supervisors_current_immutable_snapshot() {
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();

    let (status, body) = request(&runtime, "GET", "/api/v1/status").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, serde_json::to_value(runtime.snapshot()).unwrap());

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn lifecycle_routes_delegate_to_the_single_runtime_handle() {
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();

    let (status, started) = request(&runtime, "POST", "/api/v1/runtime/start").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(started["pipeline"]["state"], "running");
    assert_eq!(started["pipeline"]["epoch"], 1);

    let (status, restarted) = request(&runtime, "POST", "/api/v1/runtime/restart").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(restarted["pipeline"]["state"], "running");
    assert_eq!(restarted["pipeline"]["epoch"], 2);

    let (status, stopped) = request(&runtime, "POST", "/api/v1/runtime/stop").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(stopped["pipeline"]["state"], "stopped");
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn emergency_stop_is_exposed_without_a_daemon_shutdown_route() {
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    request(&runtime, "POST", "/api/v1/runtime/start").await;

    let (status, stopped) = request(&runtime, "POST", "/api/v1/runtime/emergency-stop").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(stopped["pipeline"]["state"], "stopped");
    assert_eq!(stopped["subsystems"]["device"]["state"], "unavailable");

    let (status, _) = request(&runtime, "POST", "/api/v1/runtime/shutdown-daemon").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn closed_supervisor_returns_a_stable_service_unavailable_error() {
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    runtime.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("join supervisor");

    let (status, body) = request(&runtime, "POST", "/api/v1/runtime/start").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["code"], "supervisor_closed");
    assert!(
        body["message"]
            .as_str()
            .is_some_and(|message| !message.is_empty())
    );
}

#[tokio::test]
async fn internal_pipeline_start_failure_is_not_misreported_as_a_client_conflict() {
    #[derive(Debug)]
    struct FixedClock;

    impl Clock for FixedClock {
        fn now(&self) -> MonotonicNanos {
            MonotonicNanos(1_000_000_000)
        }
    }

    let clock: Arc<dyn Clock> = Arc::new(FixedClock);
    let device: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
    let dependencies = RuntimeDependencies::new(
        clock,
        device,
        PipelineConfig {
            output_interval_ms: 0,
            ..PipelineConfig::default()
        },
    );
    let (supervisor, runtime) = RuntimeSupervisor::spawn(dependencies);

    let (status, body) = request(&runtime, "POST", "/api/v1/runtime/start").await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body["code"], "pipeline_rejected");

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn events_websocket_sends_initial_and_changed_supervisor_snapshots() {
    use futures_util::StreamExt;

    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let address = listener.local_addr().expect("server address");
    let app = build_control_router(runtime.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve control API");
    });
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{address}/api/v1/events"))
        .await
        .expect("connect events socket");

    let initial = socket.next().await.expect("initial frame").expect("frame");
    let initial: Value =
        serde_json::from_str(initial.to_text().expect("text frame")).expect("initial event JSON");
    assert_eq!(initial["kind"], "runtime_snapshot");
    assert_eq!(initial["snapshot"]["pipeline"]["state"], "stopped");

    runtime.start().await.expect("start runtime");
    let changed = loop {
        let changed = socket.next().await.expect("changed frame").expect("frame");
        let changed: Value = serde_json::from_str(changed.to_text().expect("text frame"))
            .expect("changed event JSON");
        if changed["snapshot"]["pipeline"]["state"] == "running" {
            break changed;
        }
    };
    assert_eq!(changed["snapshot"]["pipeline"]["state"], "running");
    assert_eq!(changed["snapshot"]["pipeline"]["epoch"], 1);

    socket.close(None).await.expect("close socket");
    server.abort();
    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn daemon_shutdown_closes_events_websocket_and_releases_axum_drain() {
    use futures_util::StreamExt;

    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let address = listener.local_addr().expect("server address");
    let app = build_control_router_with_shutdown(runtime.clone(), Some(shutdown_rx.clone()));
    let mut graceful_rx = shutdown_rx;
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                if !*graceful_rx.borrow() {
                    let _ = graceful_rx.changed().await;
                }
            })
            .await
            .expect("serve control API");
    });
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{address}/api/v1/events"))
        .await
        .expect("connect events socket");
    socket.next().await.expect("initial frame").expect("frame");

    shutdown_tx.send_replace(true);
    let closed = tokio::time::timeout(std::time::Duration::from_secs(1), socket.next())
        .await
        .expect("events socket closes on daemon shutdown");
    assert!(
        matches!(
            closed,
            None | Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_))) | Some(Err(_))
        ),
        "unexpected frame after shutdown: {closed:?}"
    );
    tokio::time::timeout(std::time::Duration::from_secs(1), server)
        .await
        .expect("Axum graceful drain completes")
        .expect("server task");

    shutdown(supervisor, &runtime).await;
}
