use std::sync::Arc;

use axum::{
    Router,
    extract::{
        Query, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    response::Response,
    routing::get,
};
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use novasight_core::OperationalSnapshot;

use crate::{ApiState, dto::RuntimeStateResponse};

const RUNTIME_SNAPSHOT_KIND: &str = "runtime_snapshot";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StatusTopic {
    Full,
    Summary,
    Capture,
    Infer,
    Control,
    Latency,
}

impl StatusTopic {
    pub fn normalize(value: Option<&str>) -> Self {
        match value
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "summary" => Self::Summary,
            "capture" => Self::Capture,
            "infer" => Self::Infer,
            "control" => Self::Control,
            "latency" => Self::Latency,
            _ => Self::Full,
        }
    }

    const fn sends_full_state(self) -> bool {
        matches!(
            self,
            Self::Full | Self::Infer | Self::Control | Self::Latency
        )
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeStatusFrame {
    kind: &'static str,
    topic: StatusTopic,
    full: bool,
    state: RuntimeStateResponse,
}

impl RuntimeStatusFrame {
    pub fn new(topic: StatusTopic, state: RuntimeStateResponse) -> Self {
        Self {
            kind: RUNTIME_SNAPSHOT_KIND,
            topic,
            full: topic.sends_full_state(),
            state,
        }
    }

    pub fn full(state: RuntimeStateResponse) -> Self {
        Self::new(StatusTopic::Full, state)
    }
}

#[derive(Debug, Default, Deserialize)]
struct StatusQuery {
    topic: Option<String>,
}

pub(crate) fn router() -> Router<ApiState> {
    Router::new().route("/ws/status", get(upgrade_status))
}

async fn upgrade_status(
    websocket: WebSocketUpgrade,
    Query(query): Query<StatusQuery>,
    State(state): State<ApiState>,
) -> Response {
    let topic = StatusTopic::normalize(query.topic.as_deref());
    let receiver = state.runtime.subscribe();
    websocket.on_upgrade(move |socket| stream_status(socket, receiver, topic))
}

async fn stream_status(
    socket: WebSocket,
    mut snapshots: watch::Receiver<Arc<OperationalSnapshot>>,
    topic: StatusTopic,
) {
    let (mut outbound, mut inbound) = socket.split();
    let Some(initial) = current_message(&mut snapshots, topic) else {
        return;
    };
    if !send_while_receiving(&mut outbound, &mut inbound, initial).await {
        return;
    }

    loop {
        tokio::select! {
            changed = snapshots.changed() => {
                if changed.is_err() {
                    return;
                }
                let Some(message) = current_message(&mut snapshots, topic) else {
                    return;
                };
                if !send_while_receiving(&mut outbound, &mut inbound, message).await {
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

fn current_message(
    snapshots: &mut watch::Receiver<Arc<OperationalSnapshot>>,
    topic: StatusTopic,
) -> Option<Message> {
    let frame = project_current_with(snapshots, |snapshot| {
        RuntimeStatusFrame::new(topic, RuntimeStateResponse::from(snapshot))
    });
    serde_json::to_string(&frame)
        .ok()
        .map(|payload| Message::Text(payload.into()))
}

fn project_current_with<T>(
    snapshots: &mut watch::Receiver<Arc<OperationalSnapshot>>,
    project: impl FnOnce(&OperationalSnapshot) -> T,
) -> T {
    let snapshot = {
        let borrowed = snapshots.borrow_and_update();
        Arc::clone(&borrowed)
    };
    project(snapshot.as_ref())
}

async fn send_while_receiving<S, R, E>(outbound: &mut S, inbound: &mut R, message: Message) -> bool
where
    S: Sink<Message> + Unpin,
    R: Stream<Item = Result<Message, E>> + Unpin,
{
    let send = outbound.send(message);
    tokio::pin!(send);

    loop {
        tokio::select! {
            result = &mut send => return result.is_ok(),
            incoming = inbound.next() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return false,
                    Some(Ok(_)) => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::Future,
        pin::Pin,
        sync::Arc,
        task::{Context, Poll},
        thread,
        time::Duration,
    };

    use futures_util::{Sink, Stream};
    use novasight_core::{OperationalSnapshot, RunIntent, RuntimeEpoch, RuntimePhase};
    use tokio::sync::{oneshot, watch};

    use super::{Message, project_current_with, send_while_receiving};

    fn snapshot(epoch: u64) -> OperationalSnapshot {
        OperationalSnapshot {
            phase: RuntimePhase::Running,
            run_intent: RunIntent::Running,
            epoch: Some(RuntimeEpoch(epoch)),
            running: true,
            source: "replay".to_owned(),
            last_generation: None,
            processed_batches: 0,
            device_receipts: 0,
            fatal_error: None,
        }
    }

    #[test]
    fn projection_drops_watch_guard_before_running_converter() {
        let (sender, mut receiver) = watch::channel(Arc::new(snapshot(1)));
        let (projection_started_tx, projection_started_rx) = std::sync::mpsc::channel();
        let (release_projection_tx, release_projection_rx) = std::sync::mpsc::channel();

        let projection = thread::spawn(move || {
            project_current_with(&mut receiver, |current| {
                projection_started_tx
                    .send(())
                    .expect("announce projection start");
                release_projection_rx
                    .recv()
                    .expect("release blocked projection");
                current.epoch
            })
        });
        projection_started_rx
            .recv()
            .expect("projection entered converter");

        let (publish_done_tx, publish_done_rx) = std::sync::mpsc::channel();
        let publish = thread::spawn(move || {
            sender.send_replace(Arc::new(snapshot(2)));
            publish_done_tx.send(()).expect("announce publish");
        });
        let publish_completed = publish_done_rx.recv_timeout(Duration::from_secs(1));
        release_projection_tx
            .send(())
            .expect("finish blocked projection");
        publish.join().expect("publisher thread");
        projection.join().expect("projection thread");

        assert!(
            publish_completed.is_ok(),
            "snapshot publication must not wait for DTO projection"
        );
    }

    struct PendingSink {
        polled_tx: Option<oneshot::Sender<()>>,
    }

    impl Sink<Message> for PendingSink {
        type Error = ();

        fn poll_ready(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            if let Some(polled_tx) = self.polled_tx.take() {
                let _ = polled_tx.send(());
            }
            Poll::Pending
        }

        fn start_send(self: Pin<&mut Self>, _item: Message) -> Result<(), Self::Error> {
            unreachable!("pending sink cannot accept a frame")
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
    }

    struct CloseAfterSendPending {
        send_polled_rx: oneshot::Receiver<()>,
    }

    impl Stream for CloseAfterSendPending {
        type Item = Result<Message, ()>;

        fn poll_next(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            match Pin::new(&mut self.send_polled_rx).poll(context) {
                Poll::Ready(_) => Poll::Ready(Some(Ok(Message::Close(None)))),
                Poll::Pending => Poll::Pending,
            }
        }
    }

    #[tokio::test]
    async fn blocked_outbound_send_still_observes_client_close() {
        let (send_polled_tx, send_polled_rx) = oneshot::channel();
        let mut sink = PendingSink {
            polled_tx: Some(send_polled_tx),
        };
        let mut incoming = CloseAfterSendPending { send_polled_rx };

        let connected = tokio::time::timeout(
            Duration::from_secs(1),
            send_while_receiving(&mut sink, &mut incoming, Message::Text("blocked".into())),
        )
        .await
        .expect("close must interrupt blocked outbound send");

        assert!(!connected);
    }
}
