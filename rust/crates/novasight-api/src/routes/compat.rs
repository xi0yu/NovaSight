//! Phase 1 compatibility shims for legacy NovaSight API endpoints.
//!
//! The Phase 1 Rust backend only owns the runtime lifecycle (`/api/runtime/*`)
//! and the status WebSocket. The legacy FastAPI app exposes 80+ extra
//! endpoints that the React frontend still probes on launch. Without
//! responses, the frontend logs repeated 404s and certain features stop
//! loading. This module answers the most heavily used endpoints with
//! shapes that match the Python `asdict` payloads, so the UI renders
//! cleanly. Every handler here is read-only or returns an explicit
//! `not_implemented` error for state-changing endpoints that the
//! Python backend used to own. Output gate stays closed: no live device
//! can ever be reached through this shim.

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use serde_json::{Value, json};

use crate::ApiState;

const NOT_IMPLEMENTED: &str = "not_implemented";

pub(super) fn router() -> Router<ApiState> {
    Router::new()
        .route("/api/config", get(get_config).post(post_config))
        .route("/api/config/schema", get(get_config_schema))
        .route("/api/license", get(get_license).delete(delete_license))
        .route("/api/license/activate", post(activate_license))
        .route("/api/capture/state", get(get_capture_state))
        .route(
            "/api/capture/capabilities",
            get(get_capture_capabilities).post(post_capture_capabilities),
        )
        .route("/api/capture/preview", post(post_capture_preview))
        .route("/api/capture/select", post(post_capture_select))
        .route("/api/capture/image", post(post_capture_image))
        .route("/api/capture/stop", post(post_capture_stop))
        .route("/api/crosshair", get(get_crosshair))
        .route("/api/crosshair/learn", post(post_crosshair_learn))
        .route("/api/crosshair/template", delete(delete_crosshair_template))
        .route("/api/executors", get(get_executors))
        .route("/api/executors/kmnet/connect", post(kmnet_connect))
        .route("/api/executors/kmnet/disconnect", post(kmnet_disconnect))
        .route("/api/executors/kmnet/buttons", get(kmnet_buttons))
        .route(
            "/api/executors/kmnet/diagnostic-move",
            post(kmnet_diagnostic_move),
        )
        .route(
            "/api/executors/kmnet/diagnostic-circle",
            post(kmnet_diagnostic_circle),
        )
        .route("/api/models/projects", get(get_model_projects))
        .route("/api/models/catalog", get(get_model_catalog))
        .route("/api/models/jobs", get(get_model_jobs))
        .route("/api/models/jobs/list", get(get_model_jobs_list))
        .route("/api/motion/runtime", get(get_motion_runtime))
        .route("/api/motion/sessions", get(get_motion_sessions))
        .route("/api/motion/profiles", get(get_motion_profiles))
}

fn not_implemented(action: &str) -> Response {
    let body = Json(json!({
        "code": NOT_IMPLEMENTED,
        "detail": {
            "message": format!(
                "the Phase 1 Rust backend is a replay-only compatibility shim; \
                 `{action}` is not yet wired to a Rust handler. \
                 Start the Python FastAPI app for full functionality."
            )
        }
    }));
    (StatusCode::NOT_IMPLEMENTED, body).into_response()
}

fn not_connected_executor(name: &str) -> Response {
    let body = Json(json!({
        "code": "executor_unavailable",
        "detail": {
            "message": format!("{name} is not available in the Phase 1 replay backend")
        }
    }));
    (StatusCode::SERVICE_UNAVAILABLE, body).into_response()
}

async fn get_config(State(_state): State<ApiState>) -> Json<Value> {
    Json(json!({
        "version": 0,
        "capture": { "device": "", "profile": null, "available": false },
        "inference": { "available": false, "configured": false, "running": false },
        "runtime": { "freshness_threshold_ms": 55.0, "drop_stale_batches": true, "consume_latest_only": true },
        "control": {
            "active_algorithm": "replay_only",
            "algorithms": {},
            "aim": { "role_y_ratios": { "head": 0.0, "body": 0.0, "other": 0.0 } }
        },
        "consumers": { "recording_format": "csv" },
        "legacy": {}
    }))
}

async fn post_config() -> Response {
    not_implemented("POST /api/config")
}

async fn get_config_schema() -> Json<Value> {
    Json(json!({
        "version": 0,
        "values": {},
        "sections": []
    }))
}

async fn get_license() -> Json<Value> {
    Json(json!({
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
    }))
}

async fn delete_license() -> Json<Value> {
    get_license().await
}

async fn activate_license() -> Response {
    not_implemented("POST /api/license/activate")
}

async fn get_capture_state() -> Json<Value> {
    Json(json!({
        "available": false,
        "device": "",
        "profile": null,
        "reason": "capture pipeline lives in the Python backend; Phase 1 Rust replays fixtures only"
    }))
}

async fn get_capture_capabilities() -> Json<Value> {
    Json(json!({
        "available": false,
        "devices": []
    }))
}

async fn post_capture_capabilities() -> Response {
    not_implemented("POST /api/capture/capabilities")
}

async fn post_capture_preview() -> Response {
    not_implemented("POST /api/capture/preview")
}

async fn post_capture_select() -> Response {
    not_implemented("POST /api/capture/select")
}

async fn post_capture_image() -> Response {
    not_implemented("POST /api/capture/image")
}

async fn post_capture_stop() -> Json<Value> {
    Json(json!({ "stopped": true, "reason": "no capture pipeline is active" }))
}

async fn get_crosshair() -> Json<Value> {
    Json(json!({
        "configured": false,
        "template_ready": false,
        "screenshot_available": false
    }))
}

async fn post_crosshair_learn() -> Response {
    not_implemented("POST /api/crosshair/learn")
}

async fn delete_crosshair_template() -> Json<Value> {
    Json(json!({ "cleared": true }))
}

async fn get_executors() -> Json<Value> {
    Json(json!({
        "selected": "dry_run",
        "executors": {
            "dry_run": { "available": true },
            "kmnet": { "available": false }
        }
    }))
}

async fn kmnet_connect() -> Response {
    not_connected_executor("kmnet")
}

async fn kmnet_disconnect() -> Response {
    not_connected_executor("kmnet")
}

async fn kmnet_buttons() -> Json<Value> {
    Json(json!({ "buttons": [] }))
}

async fn kmnet_diagnostic_move() -> Response {
    not_connected_executor("kmnet")
}

async fn kmnet_diagnostic_circle() -> Response {
    not_connected_executor("kmnet")
}

async fn get_model_projects() -> Json<Value> {
    Json(json!({ "projects": [] }))
}

async fn get_model_catalog() -> Json<Value> {
    Json(json!({ "artifacts": [] }))
}

async fn get_model_jobs() -> Json<Value> {
    Json(json!({ "jobs": [] }))
}

async fn get_model_jobs_list() -> Json<Value> {
    Json(json!({ "jobs": [] }))
}

async fn get_motion_runtime() -> Json<Value> {
    Json(json!({
        "active_profile": null,
        "builtin_active": false
    }))
}

async fn get_motion_sessions() -> Json<Value> {
    Json(json!({ "sessions": [] }))
}

async fn get_motion_profiles() -> Json<Value> {
    Json(json!({ "profiles": [] }))
}
