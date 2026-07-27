use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use futures_util::StreamExt;
use novasight_api::build_control_router_with_control_plane;
use novasight_runtime::{PipelineState, RuntimeSupervisor};
use novasight_store::license::{FileLicenseRepository, LicensePolicy};
use serde_json::Value;
use tower::ServiceExt;

const TEST_KEY: &str = "NOVASIGHT-TEST-MAX-ACCESS-2026";
const PUBLIC_KEY: &str = include_str!("../../../testdata/license-public.pem");
const SIGNED_KEY: &str = include_str!("../../../testdata/license-signed.key");
static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let unique = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "novasight-license-api-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

async fn json_response(
    app: axum::Router,
    method: &str,
    path: &str,
    body: Body,
) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method(method)
                .uri(path)
                .header("content-type", "application/json")
                .body(body)
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

#[tokio::test]
async fn license_gate_blocks_runtime_until_real_activation_and_clear() {
    let directory = TestDirectory::new();
    let license = FileLicenseRepository::new(
        directory.0.join("license.json"),
        LicensePolicy::new(true, None),
    );
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app =
        build_control_router_with_control_plane(runtime.clone(), None, license, None, false, None);

    let (status, body) = json_response(app.clone(), "GET", "/api/license", Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["configured"], false);
    assert_eq!(body["valid"], false);

    let (status, _) = json_response(app.clone(), "DELETE", "/api/license", Body::empty()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let cors_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/runtime/start")
                .header("origin", "http://localhost:5173")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cors_response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        cors_response
            .headers()
            .get("access-control-allow-origin")
            .unwrap(),
        "http://localhost:5173"
    );

    let (status, body) =
        json_response(app.clone(), "POST", "/api/runtime/start", Body::empty()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["code"], "LICENSE_REQUIRED");
    assert_eq!(body["detail"], "license required");
    assert_eq!(body["message"], "a valid license is required");
    assert_eq!(body["license"]["valid"], false);
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);

    let (status, body) = json_response(
        app.clone(),
        "PUT",
        "/api/license",
        Body::from(format!(r#"{{"key":"{TEST_KEY}"}}"#)),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["valid"], true);
    assert_eq!(body["tier"], "test_max");

    let (status, body) =
        json_response(app.clone(), "POST", "/api/runtime/start", Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["running"], true);

    let (status, body) = json_response(app.clone(), "DELETE", "/api/license", Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["configured"], false);
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);

    let (status, body) = json_response(app, "POST", "/api/runtime/stop", Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["running"], false);
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
}

#[tokio::test]
async fn unlicensed_status_websocket_closes_with_the_python_compatible_code() {
    let directory = TestDirectory::new();
    let license = FileLicenseRepository::new(
        directory.0.join("license.json"),
        LicensePolicy::new(true, None),
    );
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app =
        build_control_router_with_control_plane(runtime.clone(), None, license, None, false, None);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{address}/ws/status"))
        .await
        .unwrap();
    let close = socket.next().await.unwrap().unwrap();
    let tokio_tungstenite::tungstenite::Message::Close(Some(frame)) = close else {
        panic!("expected an authorization close frame, got {close:?}");
    };
    assert_eq!(u16::from(frame.code), 4401);
    assert_eq!(frame.reason, "license required");

    server.abort();
    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
}

#[tokio::test]
async fn license_gate_enforces_features_on_the_server() {
    let directory = TestDirectory::new();
    let license = FileLicenseRepository::new(
        directory.0.join("license.json"),
        LicensePolicy::new(false, Some(PUBLIC_KEY.to_owned())),
    );
    license.activate(SIGNED_KEY).unwrap();
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app =
        build_control_router_with_control_plane(runtime.clone(), None, license, None, true, None);

    let (status, body) =
        json_response(app.clone(), "GET", "/api/runtime/state", Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["running"], false);

    let (status, body) =
        json_response(app.clone(), "POST", "/api/runtime/start", Body::empty()).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["code"], "LICENSE_FEATURE_REQUIRED");
    assert_eq!(
        body["message"],
        "license feature hardware_control is required"
    );
    assert_eq!(body["required_feature"], "hardware_control");
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);

    let (status, body) = json_response(app.clone(), "GET", "/api/executors", Body::empty()).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["code"], "LICENSE_FEATURE_REQUIRED");
    assert_eq!(body["required_feature"], "hardware_control");

    let (status, body) = json_response(
        app,
        "POST",
        "/api/config",
        Body::from(r#"{"section":"server","key":"port","value":6000}"#),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["required_feature"], "config_write");

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
}
