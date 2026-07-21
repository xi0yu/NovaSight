use std::{
    io::{Read, Write},
    net::TcpStream,
    sync::Arc,
    time::Duration,
};

use novasight_api::{
    ApiState, build_router,
    dto::RuntimeStateResponse,
    websocket::status::{RuntimeStatusFrame, StatusTopic},
};
use novasight_core::{
    Generation, OperationalSnapshot, RunIntent, RuntimeDependencies, RuntimeEpoch, RuntimeManager,
    RuntimePhase,
};
use serde_json::Value;
use tokio::sync::watch;

fn snapshot(epoch: Option<u64>, generation: Option<u64>) -> OperationalSnapshot {
    OperationalSnapshot {
        phase: if epoch.is_some() {
            RuntimePhase::Running
        } else {
            RuntimePhase::Stopped
        },
        run_intent: if epoch.is_some() {
            RunIntent::Running
        } else {
            RunIntent::Stopped
        },
        epoch: epoch.map(RuntimeEpoch),
        running: epoch.is_some(),
        source: "replay".to_owned(),
        last_generation: generation.map(Generation),
        processed_batches: generation.unwrap_or_default(),
        device_receipts: generation.unwrap_or_default(),
        fatal_error: None,
    }
}

fn stopped_state() -> RuntimeStateResponse {
    RuntimeStateResponse::from(&snapshot(None, None))
}

#[test]
fn status_frame_matches_frontend_envelope_for_every_topic() {
    let cases = [
        ("full", "full", true),
        ("summary", "summary", false),
        ("capture", "capture", false),
        ("infer", "infer", true),
        ("control", "control", true),
        ("latency", "latency", true),
        ("unknown", "full", true),
        (" SUMMARY ", "summary", false),
    ];

    for (requested, expected_topic, expected_full) in cases {
        let topic = StatusTopic::normalize(Some(requested));
        let value = serde_json::to_value(RuntimeStatusFrame::new(topic, stopped_state()))
            .expect("status frame JSON");

        assert_eq!(value["kind"], "runtime_snapshot", "requested {requested}");
        assert_eq!(value["topic"], expected_topic, "requested {requested}");
        assert_eq!(value["full"], expected_full, "requested {requested}");
        assert!(value["state"].is_object(), "requested {requested}");
        assert_eq!(value.as_object().expect("frame object").len(), 4);
    }

    let default = serde_json::to_value(RuntimeStatusFrame::full(stopped_state()))
        .expect("default full frame JSON");
    assert_eq!(default["topic"], "full");
    assert_eq!(default["full"], true);
}

#[tokio::test]
async fn slow_watch_receiver_skips_intermediate_snapshots() {
    let (sender, mut receiver) = watch::channel(Arc::new(snapshot(Some(1), Some(1))));

    sender
        .send(Arc::new(snapshot(Some(2), Some(4))))
        .expect("intermediate snapshot");
    sender
        .send(Arc::new(snapshot(Some(3), Some(9))))
        .expect("newest snapshot");

    receiver.changed().await.expect("latest snapshot available");
    let newest = receiver.borrow_and_update().clone();
    assert_eq!(newest.epoch, Some(RuntimeEpoch(3)));
    assert_eq!(newest.last_generation, Some(Generation(9)));
    assert!(!receiver.has_changed().expect("sender remains open"));
}

#[tokio::test]
async fn status_route_upgrades_and_sends_current_snapshot_immediately() {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture());
    runtime.start().await.expect("start runtime");
    let observed_runtime = runtime.clone();
    let app = build_router(ApiState::new(runtime));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let address = listener.local_addr().expect("test server address");
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve test router");
    });

    let payload = tokio::time::timeout(
        Duration::from_secs(3),
        tokio::task::spawn_blocking(move || receive_first_text_frame(address, "capture")),
    )
    .await
    .expect("initial WebSocket frame timeout")
    .expect("blocking client task")
    .expect("WebSocket exchange");
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        observed_runtime.snapshot().running,
        "closing one client must not stop or mutate runtime"
    );
    observed_runtime.stop().await.expect("stop test runtime");
    server.abort();

    let frame: Value = serde_json::from_str(&payload).expect("status frame JSON");
    assert_eq!(frame["kind"], "runtime_snapshot");
    assert_eq!(frame["topic"], "capture");
    assert_eq!(frame["full"], false);
    assert_eq!(frame["state"]["running"], true);
    assert_eq!(frame["state"]["pipeline"]["epoch"], 1);
    assert_eq!(frame["state"]["source"], "replay");
}

fn receive_first_text_frame(address: std::net::SocketAddr, topic: &str) -> std::io::Result<String> {
    let mut stream = TcpStream::connect(address)?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    write!(
        stream,
        "GET /ws/status?topic={topic} HTTP/1.1\r\n\
         Host: {address}\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
         Sec-WebSocket-Version: 13\r\n\r\n"
    )?;
    stream.flush()?;

    let mut response_headers = Vec::new();
    while !response_headers.ends_with(b"\r\n\r\n") {
        let mut byte = [0_u8; 1];
        stream.read_exact(&mut byte)?;
        response_headers.push(byte[0]);
    }
    let response_headers = String::from_utf8_lossy(&response_headers);
    if !response_headers.starts_with("HTTP/1.1 101") {
        return Err(std::io::Error::other(format!(
            "expected WebSocket upgrade, got {response_headers}"
        )));
    }

    let mut header = [0_u8; 2];
    stream.read_exact(&mut header)?;
    if header[0] & 0x0f != 0x01 {
        return Err(std::io::Error::other("first WebSocket frame is not text"));
    }
    if header[1] & 0x80 != 0 {
        return Err(std::io::Error::other("server frame must not be masked"));
    }

    let payload_length = match header[1] & 0x7f {
        length @ 0..=125 => u64::from(length),
        126 => {
            let mut extended = [0_u8; 2];
            stream.read_exact(&mut extended)?;
            u64::from(u16::from_be_bytes(extended))
        }
        127 => {
            let mut extended = [0_u8; 8];
            stream.read_exact(&mut extended)?;
            u64::from_be_bytes(extended)
        }
        _ => unreachable!("WebSocket payload length marker is seven bits"),
    };
    let mut payload = vec![0_u8; usize::try_from(payload_length).expect("frame fits in memory")];
    stream.read_exact(&mut payload)?;

    stream.write_all(&[0x88, 0x80, 0, 0, 0, 0])?;
    String::from_utf8(payload).map_err(std::io::Error::other)
}
