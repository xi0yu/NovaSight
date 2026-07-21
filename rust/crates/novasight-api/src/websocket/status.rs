use axum::{
    Router,
    extract::{
        Query, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    response::Response,
    routing::get,
};
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
    mut socket: WebSocket,
    mut receiver: watch::Receiver<std::sync::Arc<OperationalSnapshot>>,
    topic: StatusTopic,
) {
    if send_current(&mut socket, &mut receiver, topic)
        .await
        .is_err()
    {
        return;
    }

    loop {
        tokio::select! {
            changed = receiver.changed() => {
                if changed.is_err()
                    || send_current(&mut socket, &mut receiver, topic).await.is_err()
                {
                    return;
                }
            }
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return,
                    Some(Ok(_)) => {}
                }
            }
        }
    }
}

async fn send_current(
    socket: &mut WebSocket,
    receiver: &mut watch::Receiver<std::sync::Arc<OperationalSnapshot>>,
    topic: StatusTopic,
) -> Result<(), ()> {
    let state = {
        let snapshot = receiver.borrow_and_update();
        RuntimeStateResponse::from(snapshot.as_ref())
    };
    let payload = serde_json::to_string(&RuntimeStatusFrame::new(topic, state)).map_err(|_| ())?;
    socket
        .send(Message::Text(payload.into()))
        .await
        .map_err(|_| ())
}
