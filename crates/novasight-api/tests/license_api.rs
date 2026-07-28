use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::{Body, to_bytes};
use axum::http::{HeaderMap, Request, StatusCode, header::SET_COOKIE};
use futures_util::StreamExt;
use novasight_api::build_control_router_with_control_plane;
use novasight_runtime::{PipelineState, RuntimeSupervisor};
use novasight_store::license::{FileLicenseRepository, LicensePolicy};
use serde_json::Value;
use tower::ServiceExt;

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
    let (status, _, body) = json_response_with_cookie(app, method, path, body, None).await;
    (status, body)
}

async fn json_response_with_cookie(
    app: axum::Router,
    method: &str,
    path: &str,
    body: Body,
    cookie: Option<&str>,
) -> (StatusCode, HeaderMap, Value) {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json");
    if let Some(cookie) = cookie {
        request = request.header("cookie", cookie);
    }
    let response = app.oneshot(request.body(body).unwrap()).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, headers, serde_json::from_slice(&body).unwrap())
}

fn session_cookie(headers: &HeaderMap) -> String {
    let set_cookie = headers
        .get(SET_COOKIE)
        .expect("license response sets a session cookie")
        .to_str()
        .unwrap();
    assert!(set_cookie.contains("HttpOnly"));
    assert!(set_cookie.contains("SameSite=Strict"));
    set_cookie.split(';').next().unwrap().to_owned()
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
    assert_eq!(body["temporary_access_supported"], true);

    let (status, body) = json_response(app.clone(), "DELETE", "/api/license", Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["configured"], false);

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

    let (status, headers, body) = json_response_with_cookie(
        app.clone(),
        "POST",
        "/api/license/temporary",
        Body::from("{}"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["supported"], true);
    assert_eq!(body["granted"], true);
    assert_eq!(body["status"]["valid"], true);
    assert_eq!(body["status"]["tier"], "temporary");
    let cookie = session_cookie(&headers);

    let (status, body) =
        json_response(app.clone(), "POST", "/api/runtime/start", Body::empty()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["code"], "LICENSE_SESSION_REQUIRED");

    let (status, _, body) = json_response_with_cookie(
        app.clone(),
        "POST",
        "/api/runtime/start",
        Body::empty(),
        Some(&cookie),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["running"], true);

    let (status, headers, body) = json_response_with_cookie(
        app.clone(),
        "DELETE",
        "/api/license",
        Body::empty(),
        Some(&cookie),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["configured"], false);
    assert!(
        headers
            .get(SET_COOKIE)
            .unwrap()
            .to_str()
            .unwrap()
            .contains("Max-Age=0")
    );
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);

    let (status, body) = json_response(app, "POST", "/api/runtime/stop", Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["running"], false);
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
}

#[tokio::test]
async fn debug_license_status_keeps_temporary_access_available_when_formal_storage_is_broken() {
    let directory = TestDirectory::new();
    let blocked_parent = directory.0.join("formal-license-parent");
    fs::write(&blocked_parent, "this is a file, not a directory").unwrap();
    let license = FileLicenseRepository::new(
        blocked_parent.join("license.json"),
        LicensePolicy::new(true, None),
    );
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app =
        build_control_router_with_control_plane(runtime.clone(), None, license, None, false, None);

    let (status, body) = json_response(app, "GET", "/api/license", Body::empty()).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["configured"], false);
    assert_eq!(body["valid"], false);
    assert_eq!(body["temporary_access_supported"], true);
    assert!(
        body["message"]
            .as_str()
            .unwrap()
            .contains("formal license storage is unavailable")
    );

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
}

#[tokio::test]
async fn temporary_license_policy_rejection_is_an_explicit_business_response() {
    let directory = TestDirectory::new();
    let license = FileLicenseRepository::new(
        directory.0.join("license.json"),
        LicensePolicy::new(false, None),
    );
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app =
        build_control_router_with_control_plane(runtime.clone(), None, license, None, false, None);

    let (status, body) =
        json_response(app, "POST", "/api/license/temporary", Body::from("{}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["supported"], false);
    assert_eq!(body["granted"], false);
    assert_eq!(body["status"]["temporary_access_supported"], false);
    assert_eq!(body["status"]["valid"], false);
    assert!(
        body["status"]["message"]
            .as_str()
            .unwrap()
            .contains("unavailable in this build")
    );

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

    let (status, headers, body) =
        json_response_with_cookie(app.clone(), "GET", "/api/license", Body::empty(), None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["valid"], true);
    let cookie = session_cookie(&headers);

    let (status, _, body) = json_response_with_cookie(
        app.clone(),
        "GET",
        "/api/runtime/state",
        Body::empty(),
        Some(&cookie),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["running"], false);

    let (status, _, body) = json_response_with_cookie(
        app.clone(),
        "POST",
        "/api/runtime/start",
        Body::empty(),
        Some(&cookie),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["code"], "LICENSE_FEATURE_REQUIRED");
    assert_eq!(
        body["message"],
        "license feature hardware_control is required"
    );
    assert_eq!(body["required_feature"], "hardware_control");
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);

    let (status, _, body) = json_response_with_cookie(
        app.clone(),
        "GET",
        "/api/executors",
        Body::empty(),
        Some(&cookie),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["code"], "LICENSE_FEATURE_REQUIRED");
    assert_eq!(body["required_feature"], "hardware_control");

    let (status, _, body) = json_response_with_cookie(
        app,
        "POST",
        "/api/config",
        Body::from(r#"{"section":"server","key":"port","value":6000}"#),
        Some(&cookie),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["required_feature"], "config_write");

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
}
