//! Contract tests for the Phase 1 compatibility shim. The Python FastAPI
//! app exposes endpoints that the React frontend still probes; the shim
//! answers them with shapes that match the Python `asdict` payloads so the
//! UI renders cleanly without a live Python process. These tests pin the
//! shapes so a future change cannot silently break the frontend.

use std::time::Duration;

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
    response::Response,
};
use novasight_api::{ApiState, build_router};
use novasight_core::{RuntimeDependencies, RuntimeManager};
use serde_json::{Value, json};
use tower::ServiceExt;

fn app() -> axum::Router {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture(Duration::ZERO));
    build_router(ApiState::new(runtime))
}

async fn send(app: axum::Router, request: Request<Body>) -> Response {
    app.oneshot(request).await.expect("router response")
}

async fn response_json(response: Response) -> Value {
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    serde_json::from_slice(&bytes).expect("JSON response")
}

async fn get_json(app: axum::Router, path: &str) -> Value {
    let response = send(
        app,
        Request::get(path).body(Body::empty()).expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    response_json(response).await
}

#[tokio::test]
async fn license_status_reports_unconfigured_compatibly() {
    let body = get_json(app(), "/api/license").await;

    assert_eq!(
        body,
        json!({
            "configured": false,
            "valid": false,
            "fingerprint": "",
            "tier": "",
            "features": [],
            "license_id": "",
            "created_at": null,
            "activated_at": null,
            "expires_at": null,
            "duration_value": null,
            "duration_unit": "",
            "updated_at": null,
            "message": "Phase 1 Rust backend does not enforce licensing"
        })
    );
}

#[tokio::test]
async fn license_delete_returns_the_unconfigured_shape() {
    let response = send(
        app(),
        Request::delete("/api/license")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body["configured"], json!(false));
    assert_eq!(body["valid"], json!(false));
}

#[tokio::test]
async fn config_returns_an_empty_section_summary() {
    let body = get_json(app(), "/api/config").await;

    assert_eq!(body["version"], json!(0));
    assert_eq!(body["capture"]["available"], json!(false));
    assert_eq!(body["inference"]["available"], json!(false));
    assert_eq!(body["runtime"]["drop_stale_batches"], json!(true));
    assert_eq!(body["control"]["active_algorithm"], json!("replay_only"));
}

#[tokio::test]
async fn config_schema_returns_zero_fields() {
    let body = get_json(app(), "/api/config/schema").await;

    assert_eq!(body["version"], json!(0));
    assert_eq!(body["values"], json!({}));
    assert_eq!(body["sections"], json!([]));
}

#[tokio::test]
async fn executors_report_dry_run_only_with_kmnet_offline() {
    let body = get_json(app(), "/api/executors").await;

    assert_eq!(body["selected"], json!("dry_run"));
    assert_eq!(body["executors"]["dry_run"]["available"], json!(true));
    assert_eq!(body["executors"]["kmnet"]["available"], json!(false));
}

#[tokio::test]
async fn capture_state_reports_unavailable_with_reason() {
    let body = get_json(app(), "/api/capture/state").await;

    assert_eq!(body["available"], json!(false));
    assert!(
        body["reason"]
            .as_str()
            .unwrap_or_default()
            .contains("Python backend"),
        "expected a reason explaining the limitation, got: {body}"
    );
}

#[tokio::test]
async fn kmnet_connect_returns_503_service_unavailable() {
    let response = send(
        app(),
        Request::post("/api/executors/kmnet/connect")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = response_json(response).await;
    assert_eq!(body["code"], json!("executor_unavailable"));
}

#[tokio::test]
async fn license_activate_returns_not_implemented() {
    let response = send(
        app(),
        Request::post("/api/license/activate")
            .header("content-type", "application/json")
            .body(Body::from("{\"key\":\"test\"}"))
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let body = response_json(response).await;
    assert_eq!(body["code"], json!("not_implemented"));
}

#[tokio::test]
async fn config_post_returns_not_implemented() {
    let response = send(
        app(),
        Request::post("/api/config")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
}

#[tokio::test]
async fn model_lists_are_empty() {
    assert_eq!(
        get_json(app(), "/api/models/projects").await,
        json!({ "projects": [] })
    );
    assert_eq!(
        get_json(app(), "/api/models/catalog").await,
        json!({ "artifacts": [] })
    );
    assert_eq!(
        get_json(app(), "/api/models/jobs").await,
        json!({ "jobs": [] })
    );
    assert_eq!(
        get_json(app(), "/api/models/jobs/list").await,
        json!({ "jobs": [] })
    );
}

#[tokio::test]
async fn motion_lists_are_empty_with_no_active_profile() {
    assert_eq!(
        get_json(app(), "/api/motion/runtime").await,
        json!({ "active_profile": null, "builtin_active": false })
    );
    assert_eq!(
        get_json(app(), "/api/motion/sessions").await,
        json!({ "sessions": [] })
    );
    assert_eq!(
        get_json(app(), "/api/motion/profiles").await,
        json!({ "profiles": [] })
    );
}

#[tokio::test]
async fn crosshair_status_reports_unconfigured() {
    let body = get_json(app(), "/api/crosshair").await;

    assert_eq!(body["configured"], json!(false));
    assert_eq!(body["template_ready"], json!(false));
    assert_eq!(body["screenshot_available"], json!(false));
}
