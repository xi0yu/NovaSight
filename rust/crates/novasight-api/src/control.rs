use std::time::Duration;

use axum::{
    Json, Router,
    extract::{
        Query, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures_util::{Sink, Stream, StreamExt};
use novasight_runtime::{
    AppConfig, ConfigFieldUpdate, ConfigService, ConfigServiceError, ConfigUpdate, DaemonState,
    RuntimeError, RuntimeErrorKind, RuntimeHandle, RuntimeSnapshot,
};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use crate::dto::{
    CompatibilityHealth, CompatibilityRuntimeStart, CompatibilityRuntimeState,
    CompatibilityStatusFrame,
};
use crate::websocket::status::send_while_receiving;

const COMPATIBILITY_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(200);

/// Cuttlefish-style control surface: every mutation delegates to the
/// single daemon-owned RuntimeHandle and returns its immutable snapshot.
pub fn build_control_router(runtime: RuntimeHandle) -> Router {
    build_control_router_with_shutdown(runtime, None)
}

/// Build the control surface with a daemon-owned shutdown signal.
/// WebSocket upgrades observe it and leave Axum's graceful drain.
pub fn build_control_router_with_shutdown(
    runtime: RuntimeHandle,
    shutdown: impl Into<Option<watch::Receiver<bool>>>,
) -> Router {
    build_control_router_with_services(runtime, None, shutdown)
}

pub fn build_control_router_with_services(
    runtime: RuntimeHandle,
    config_service: impl Into<Option<ConfigService>>,
    shutdown: impl Into<Option<watch::Receiver<bool>>>,
) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/api/runtime/state", get(legacy_status))
        .route("/api/runtime/start", post(legacy_start))
        .route("/api/runtime/stop", post(legacy_stop))
        .route("/ws/status", get(legacy_events))
        .route("/api/v1/status", get(status))
        .route("/api/v1/config", get(config).patch(update_config))
        .route("/api/v1/runtime/start", post(start))
        .route("/api/v1/runtime/stop", post(stop))
        .route("/api/v1/runtime/restart", post(restart))
        .route("/api/v1/runtime/emergency-stop", post(emergency_stop))
        .route("/api/v1/events", get(events))
        .route("/api/config", get(config).post(update_legacy_config))
        .with_state(ControlState {
            runtime,
            config: config_service.into(),
            shutdown: shutdown.into(),
        })
        .layer(super::app::studio_cors_layer())
}

#[derive(Clone)]
struct ControlState {
    runtime: RuntimeHandle,
    config: Option<ConfigService>,
    shutdown: Option<watch::Receiver<bool>>,
}

async fn health(State(state): State<ControlState>) -> Json<CompatibilityHealth> {
    let daemon = state.runtime.snapshot().daemon.state;
    Json(CompatibilityHealth {
        ok: daemon == DaemonState::Ready,
    })
}

async fn legacy_status(State(state): State<ControlState>) -> Json<CompatibilityRuntimeState> {
    let snapshot = state.runtime.snapshot();
    Json(compatibility_state(&state, &snapshot).await)
}

async fn legacy_start(
    State(state): State<ControlState>,
) -> Result<Json<CompatibilityRuntimeStart>, ControlApiError> {
    ensure_config_effective(&state).await?;
    let snapshot = state.runtime.start().await?;
    Ok(Json(CompatibilityRuntimeStart::from(&snapshot)))
}

async fn legacy_stop(
    State(state): State<ControlState>,
) -> Result<Json<CompatibilityRuntimeState>, ControlApiError> {
    let snapshot = state.runtime.stop().await?;
    Ok(Json(compatibility_state(&state, &snapshot).await))
}

async fn compatibility_state(
    state: &ControlState,
    snapshot: &RuntimeSnapshot,
) -> CompatibilityRuntimeState {
    let config = match &state.config {
        Some(service) => Some(service.snapshot().await),
        None => None,
    };
    let effective_revision = state.config.as_ref().map(ConfigService::effective_revision);
    CompatibilityRuntimeState::new(snapshot, config.as_ref(), effective_revision)
}

async fn ensure_config_effective(state: &ControlState) -> Result<(), ControlApiError> {
    if let Some(service) = &state.config {
        service.ensure_effective().await?;
    }
    Ok(())
}

async fn config(State(state): State<ControlState>) -> Result<Json<AppConfig>, ControlApiError> {
    let service = state.config.ok_or(ControlApiError::ConfigUnavailable)?;
    Ok(Json(service.snapshot().await))
}

async fn update_config(
    State(state): State<ControlState>,
    Json(update): Json<ConfigFieldUpdate>,
) -> Result<Json<ConfigUpdate>, ControlApiError> {
    let service = state.config.ok_or(ControlApiError::ConfigUnavailable)?;
    Ok(Json(service.update_field(update).await?))
}

async fn update_legacy_config(
    State(state): State<ControlState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<ConfigUpdate>, ControlApiError> {
    let service = state.config.ok_or(ControlApiError::ConfigUnavailable)?;
    let is_field_update = payload.get("section").is_some()
        || payload.get("key").is_some()
        || payload.get("value").is_some();
    let update = if is_field_update {
        let update =
            serde_json::from_value(payload).map_err(ControlApiError::InvalidFieldUpdate)?;
        service.update_field(update).await?
    } else {
        service.replace(payload).await?
    };
    Ok(Json(update))
}

async fn status(State(state): State<ControlState>) -> Json<RuntimeSnapshot> {
    Json(state.runtime.snapshot())
}

async fn start(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    ensure_config_effective(&state).await?;
    Ok(Json(state.runtime.start().await?))
}

async fn stop(State(state): State<ControlState>) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    Ok(Json(state.runtime.stop().await?))
}

async fn restart(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    ensure_config_effective(&state).await?;
    Ok(Json(state.runtime.restart().await?))
}

async fn emergency_stop(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    Ok(Json(state.runtime.emergency_stop().await?))
}

async fn events(websocket: WebSocketUpgrade, State(state): State<ControlState>) -> Response {
    websocket.on_upgrade(move |socket| stream_events(socket, state.runtime, state.shutdown))
}

#[derive(Debug, Default, Deserialize)]
struct CompatibilityStatusQuery {
    topic: Option<String>,
}

async fn legacy_events(
    websocket: WebSocketUpgrade,
    Query(query): Query<CompatibilityStatusQuery>,
    State(state): State<ControlState>,
) -> Response {
    websocket.on_upgrade(move |socket| stream_legacy_events(socket, state, query))
}

async fn stream_legacy_events(
    socket: WebSocket,
    state: ControlState,
    query: CompatibilityStatusQuery,
) {
    let mut snapshots = state.runtime.subscribe();
    let mut shutdown = state.shutdown.clone();
    let topic = normalize_compatibility_topic(query.topic.as_deref());
    let mut heartbeat = tokio::time::interval(COMPATIBILITY_HEARTBEAT_INTERVAL);
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    heartbeat.tick().await;
    let (mut outbound, mut inbound) = socket.split();
    loop {
        let current = snapshots.borrow_and_update().clone();
        let frame = CompatibilityStatusFrame {
            kind: "runtime_snapshot",
            topic: topic.clone(),
            full: true,
            state: compatibility_state(&state, current.as_ref()).await,
        };
        let Ok(payload) = serde_json::to_string(&frame) else {
            return;
        };
        match send_or_shutdown(
            &mut outbound,
            &mut inbound,
            Message::Text(payload.into()),
            &mut shutdown,
        )
        .await
        {
            SendOutcome::Sent => {}
            SendOutcome::Closed | SendOutcome::Shutdown => return,
        }

        tokio::select! {
            changed = snapshots.changed() => {
                if changed.is_err() {
                    return;
                }
            }
            _ = heartbeat.tick() => {}
            incoming = inbound.next() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return,
                    Some(Ok(_)) => {}
                }
            }
            () = shutdown_requested(&mut shutdown) => return,
        }
    }
}

fn normalize_compatibility_topic(topic: Option<&str>) -> String {
    let topic = topic.unwrap_or_default().trim().to_ascii_lowercase();
    match topic.as_str() {
        "summary" | "capture" | "infer" | "control" | "latency" => topic,
        _ => "full".to_owned(),
    }
}

#[derive(Serialize)]
struct RuntimeEvent<'a> {
    kind: &'static str,
    snapshot: &'a RuntimeSnapshot,
}

async fn stream_events(
    socket: WebSocket,
    runtime: RuntimeHandle,
    mut shutdown: Option<watch::Receiver<bool>>,
) {
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
        match send_or_shutdown(
            &mut outbound,
            &mut inbound,
            Message::Text(payload.into()),
            &mut shutdown,
        )
        .await
        {
            SendOutcome::Sent => {}
            SendOutcome::Closed | SendOutcome::Shutdown => return,
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
            () = shutdown_requested(&mut shutdown) => return,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SendOutcome {
    Sent,
    Closed,
    Shutdown,
}

async fn send_or_shutdown<S, R, E>(
    outbound: &mut S,
    inbound: &mut R,
    message: Message,
    shutdown: &mut Option<watch::Receiver<bool>>,
) -> SendOutcome
where
    S: Sink<Message> + Unpin,
    R: Stream<Item = Result<Message, E>> + Unpin,
{
    tokio::select! {
        sent = send_while_receiving(outbound, inbound, message) => {
            if sent {
                SendOutcome::Sent
            } else {
                SendOutcome::Closed
            }
        }
        () = shutdown_requested(shutdown) => SendOutcome::Shutdown,
    }
}

async fn shutdown_requested(shutdown: &mut Option<watch::Receiver<bool>>) {
    let Some(receiver) = shutdown else {
        std::future::pending::<()>().await;
        return;
    };
    if *receiver.borrow() {
        return;
    }
    let _ = receiver.changed().await;
}

enum ControlApiError {
    Runtime(RuntimeError),
    Config(ConfigServiceError),
    ConfigUnavailable,
    InvalidFieldUpdate(serde_json::Error),
}

impl From<RuntimeError> for ControlApiError {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(error)
    }
}

impl From<ConfigServiceError> for ControlApiError {
    fn from(error: ConfigServiceError) -> Self {
        Self::Config(error)
    }
}

#[derive(Serialize)]
struct ControlErrorBody {
    code: &'static str,
    message: String,
}

impl IntoResponse for ControlApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            Self::Runtime(error) => {
                let status = match error.kind {
                    RuntimeErrorKind::InvalidPipelineState => StatusCode::CONFLICT,
                    RuntimeErrorKind::SupervisorUnavailable
                    | RuntimeErrorKind::SupervisorClosed
                    | RuntimeErrorKind::SupervisorReplyLost
                    | RuntimeErrorKind::PipelineUnavailable => StatusCode::SERVICE_UNAVAILABLE,
                    RuntimeErrorKind::PipelineRejected
                    | RuntimeErrorKind::RuntimeEpochExhausted
                    | RuntimeErrorKind::Other => StatusCode::INTERNAL_SERVER_ERROR,
                };
                (status, error.kind.code(), error.message)
            }
            Self::Config(error) => {
                let status = match error.code() {
                    "CONFIG_REVISION_CONFLICT" => StatusCode::CONFLICT,
                    "CONFIG_RESTART_REQUIRED" => StatusCode::CONFLICT,
                    "CONFIG_PARSE_ERROR"
                    | "CONFIG_VALIDATION_ERROR"
                    | "CONFIG_RESERVED_LEGACY_KEY"
                    | "CONFIG_INVALID_FIELD_TARGET"
                    | "CONFIG_FIELD_VALUE_INVALID"
                    | "CONFIG_REPLACEMENT_INVALID"
                    | "CONFIG_REPLACEMENT_REVISION_REQUIRED" => StatusCode::BAD_REQUEST,
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                };
                let code = error.code();
                (status, code, error.to_string())
            }
            Self::ConfigUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "CONFIG_SERVICE_UNAVAILABLE",
                "configuration service is not attached to this control surface".to_owned(),
            ),
            Self::InvalidFieldUpdate(error) => (
                StatusCode::BAD_REQUEST,
                "CONFIG_FIELD_UPDATE_INVALID",
                format!("expected a field update with section, key, and value: {error}"),
            ),
        };
        let body = ControlErrorBody { code, message };
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll},
        time::Duration,
    };

    use futures_util::{Sink, task::noop_waker};

    use super::*;

    struct PendingSink;

    impl Sink<Message> for PendingSink {
        type Error = ();

        fn poll_ready(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }

        fn start_send(self: Pin<&mut Self>, _item: Message) -> Result<(), Self::Error> {
            unreachable!("a permanently backpressured sink is never ready")
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }
    }

    #[tokio::test]
    async fn daemon_shutdown_interrupts_a_backpressured_websocket_send() {
        let mut outbound = PendingSink;
        let mut inbound = futures_util::stream::pending::<Result<Message, ()>>();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut shutdown = Some(shutdown_rx);
        let operation = send_or_shutdown(
            &mut outbound,
            &mut inbound,
            Message::Text("snapshot".into()),
            &mut shutdown,
        );
        tokio::pin!(operation);
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(operation.as_mut().poll(&mut context).is_pending());

        shutdown_tx.send_replace(true);
        let outcome = tokio::time::timeout(Duration::from_millis(100), operation)
            .await
            .expect("daemon shutdown must not wait for a backpressured peer");

        assert_eq!(outcome, SendOutcome::Shutdown);
    }
}
