use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};

use crate::{
    ApiState,
    dto::{RuntimeStartResponse, RuntimeStateResponse},
    error::ApiResult,
};

pub(super) fn router() -> Router<ApiState> {
    Router::new()
        .route("/api/runtime/state", get(state))
        .route("/api/runtime/start", post(start))
        .route("/api/runtime/stop", post(stop))
}

async fn state(State(state): State<ApiState>) -> Json<RuntimeStateResponse> {
    let snapshot = state.runtime.snapshot();
    Json(RuntimeStateResponse::from(snapshot.as_ref()))
}

async fn start(State(state): State<ApiState>) -> ApiResult<Json<RuntimeStartResponse>> {
    let receipt = state.runtime.start().await?;
    let running = state.runtime.snapshot().running;
    Ok(Json(RuntimeStartResponse::accepted(receipt, running)))
}

async fn stop(State(state): State<ApiState>) -> ApiResult<Json<RuntimeStateResponse>> {
    state.runtime.stop().await?;
    let snapshot = state.runtime.snapshot();
    Ok(Json(RuntimeStateResponse::from(snapshot.as_ref())))
}
