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
use futures_util::{Sink, Stream, StreamExt};
use novasight_runtime::{RuntimeError, RuntimeErrorKind, RuntimeHandle, RuntimeSnapshot};
use serde::Serialize;
use tokio::sync::watch;

use crate::websocket::status::send_while_receiving;

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
    Router::new()
        .route("/api/v1/status", get(status))
        .route("/api/v1/runtime/start", post(start))
        .route("/api/v1/runtime/stop", post(stop))
        .route("/api/v1/runtime/restart", post(restart))
        .route("/api/v1/runtime/emergency-stop", post(emergency_stop))
        .route("/api/v1/events", get(events))
        .with_state(ControlState {
            runtime,
            shutdown: shutdown.into(),
        })
        .layer(super::app::studio_cors_layer())
}

#[derive(Clone)]
struct ControlState {
    runtime: RuntimeHandle,
    shutdown: Option<watch::Receiver<bool>>,
}

async fn status(State(state): State<ControlState>) -> Json<RuntimeSnapshot> {
    Json(state.runtime.snapshot())
}

async fn start(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    Ok(Json(state.runtime.start().await?))
}

async fn stop(State(state): State<ControlState>) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
    Ok(Json(state.runtime.stop().await?))
}

async fn restart(
    State(state): State<ControlState>,
) -> Result<Json<RuntimeSnapshot>, ControlApiError> {
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
