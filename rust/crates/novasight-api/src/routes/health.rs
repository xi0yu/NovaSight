use axum::{Json, Router, routing::get};
use serde::Serialize;

use crate::ApiState;

#[derive(Serialize)]
struct HealthResponse {
    ok: bool,
}

pub(super) fn router() -> Router<ApiState> {
    Router::new().route("/healthz", get(health))
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { ok: true })
}
