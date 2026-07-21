use axum::{
    Json, Router,
    extract::{
        State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures_util::StreamExt;
use novasight_runtime::{RuntimeError, RuntimeErrorKind, RuntimeHandle, RuntimeSnapshot};
use serde::Serialize;

use crate::websocket::status::send_while_receiving;

/// Cuttlefish-style control surface: every mutation delegates to the
/// single daemon-owned RuntimeHandle and returns its immutable snapshot.
pub fn build_control_router(runtime: RuntimeHandle) -> Router {
    Router::new()
        .route("/api/v1/status", get(status))
        .route("/api/v1/runtime/start", post(start))
        .route("/api/v1/runtime/stop", post(stop))
        .route("/api/v1/runtime/restart", post(restart))
        .route("/api/v1/runtime/emergency-stop", post(emergency_stop))
        .route("/api/v1/events", get(events))
        .with_state(runtime)
        .layer(super::app::studio_cors_layer())
}

async fn status(State(runtime): State<RuntimeHandle>) -> Json<RuntimeSnapshot> {
    Json(runtime.snapshot())
}

async fn start(
    State(runtime): State<RuntimeHandle>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    Ok(Json(runtime.start().await?))
}

async fn stop(
    State(runtime): State<RuntimeHandle>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    Ok(Json(runtime.stop().await?))
}

async fn restart(
    State(runtime): State<RuntimeHandle>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    Ok(Json(runtime.restart().await?))
}

async fn emergency_stop(
    State(runtime): State<RuntimeHandle>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    Ok(Json(runtime.emergency_stop().await?))
}

async fn events(websocket: WebSocketUpgrade, State(runtime): State<RuntimeHandle>) -> Response {
    websocket.on_upgrade(move |socket| stream_events(socket, runtime))
}

#[derive(Serialize)]
struct RuntimeEvent<'a> {
    kind: &'static str,
    snapshot: &'a RuntimeSnapshot,
}

async fn stream_events(socket: WebSocket, runtime: RuntimeHandle) {
    let mut snapshots = runtime.subscribe();
    let (mut outbound, mut inbound) = socket.split();
    loop {
        let current = snapshots.borrow_and_update().clone();
        let event = RuntimeEvent {
            kind: "runtime_snapshot",
            snapshot: current.as_ref(),
        };
        let Ok(payload) = serde_json::to_string(&event) else {
            return;
        };
        if !send_while_receiving(&mut outbound, &mut inbound, Message::Text(payload.into())).await {
            return;
        }

        tokio::select! {
            changed = snapshots.changed() => {
                if changed.is_err() {
                    return;
                }
            }
            incoming = inbound.next() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return,
                    Some(Ok(_)) => {}
                }
            }
        }
    }
}

struct ControlApiError(RuntimeError);

impl From<RuntimeError> for ControlApiError {
    fn from(error: RuntimeError) -> Self {
        Self(error)
    }
}

#[derive(Serialize)]
struct ControlErrorBody {
    code: &'static str,
    message: String,
}

impl IntoResponse for ControlApiError {
    fn into_response(self) -> Response {
        let status = match self.0.kind {
            RuntimeErrorKind::InvalidPipelineState => StatusCode::CONFLICT,
            RuntimeErrorKind::SupervisorUnavailable
            | RuntimeErrorKind::SupervisorClosed
            | RuntimeErrorKind::SupervisorReplyLost
            | RuntimeErrorKind::PipelineUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            RuntimeErrorKind::PipelineRejected
            | RuntimeErrorKind::RuntimeEpochExhausted
            | RuntimeErrorKind::Other => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = ControlErrorBody {
            code: self.0.kind.code(),
            message: self.0.message,
        };
        (status, Json(body)).into_response()
    }
}
