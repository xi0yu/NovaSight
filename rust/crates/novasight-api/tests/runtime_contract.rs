use std::time::Duration;

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header::ALLOW},
    response::Response,
};
use novasight_api::{ApiState, build_router};
use novasight_core::{RuntimeDependencies, RuntimeManager};
use serde_json::{Value, json};
use tokio::sync::Barrier;
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
async fn health_matches_existing_contract() {
    let response = send(
        app(),
        Request::get("/healthz")
            .body(Body::empty())
            .expect("request"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response_json(response).await, json!({"ok": true}));
}

#[tokio::test]
async fn stopped_runtime_state_is_a_typed_replay_projection() {
    let body = get_json(app(), "/api/runtime/state").await;

    assert_eq!(
        body,
        json!({
            "running": false,
            "source": "replay",
            "active_model": null,
            "executor": {
                "selected": "dry_run",
                "executors": {
                    "dry_run": {"available": true},
                    "kmnet": {"available": false}
                }
            },
            "capture": {
                "available": false,
                "device": "replay",
                "profile": null,
                "backend": "replay",
                "fps_capture": 0.0,
                "frame_period_ms": 0.0,
                "capture_wait_ms": 0.0,
                "frames_dropped": 0,
                "preview_target_fps": 0.0,
                "preview_fps": 0.0,
                "preview_frames": 0,
                "preview_output_frames": 0,
                "preview_dropped": 0,
                "recoveries": 0,
                "last_error": null
            },
            "statistics": {
                "capture_counter": 0,
                "inference_counter": 0,
                "detection_batch_counter": 0,
                "detection_batch_consumed_counter": 0,
                "dropped_counter": 0,
                "skipped_counter": 0,
                "capture_fps": 0.0,
                "inference_fps": 0.0,
                "e2e_latency": 0.0,
                "control_observation_counter": 0
            },
            "inference": {
                "available": false,
                "configured": false,
                "running": false,
                "mode": "replay",
                "reason": "TensorRT is unavailable in Phase 1 replay mode"
            },
            "config": {"version": 1},
            "pipeline": {
                "running": false,
                "phase": "stopped",
                "run_intent": false,
                "epoch": null,
                "source": "replay",
                "last_generation": null,
                "processed_batches": 0,
                "device_receipts": 0,
                "mode": "replay"
            },
            "power_saving": {
                "enabled": false,
                "mode": "disabled",
                "run_intent": false,
                "suspended_by_policy": false,
                "running": false,
                "host_id": "",
                "target_host_id": "",
                "host_online": false,
                "heartbeat_age_ms": null,
                "auto_resume": false,
                "reason": "unavailable in Phase 1 replay mode"
            },
            "vision": {
                "available": false,
                "mode": "replay",
                "reason": "online vision hardware is unavailable in Phase 1 replay mode"
            },
            "fatal_error": null
        })
    );
}

#[tokio::test]
async fn start_delegates_to_runtime_and_returns_compatible_receipt() {
    let app = app();
    let response = send(
        app.clone(),
        Request::post("/api/runtime/start")
            .body(Body::empty())
            .expect("request"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response_json(response).await,
        json!({
            "running": true,
            "accepted": true,
            "failed": false,
            "epoch": 1,
            "operation_id": null
        })
    );
    let state = get_json(app, "/api/runtime/state").await;
    assert_eq!(state["running"], true);
    assert_eq!(state["pipeline"]["phase"], "running");
    assert_eq!(state["pipeline"]["epoch"], 1);
}

#[tokio::test]
async fn stop_waits_for_shutdown_and_returns_the_persisted_snapshot() {
    let app = app();
    let started = send(
        app.clone(),
        Request::post("/api/runtime/start")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(started.status(), StatusCode::OK);

    let stopped = send(
        app.clone(),
        Request::post("/api/runtime/stop")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(stopped.status(), StatusCode::OK);
    let stopped_body = response_json(stopped).await;
    assert_eq!(stopped_body["running"], false);
    assert_eq!(stopped_body["pipeline"]["phase"], "stopped");
    assert_eq!(stopped_body["pipeline"]["run_intent"], false);
    assert_eq!(stopped_body["pipeline"]["epoch"], 1);

    assert_eq!(
        stopped_body,
        get_json(app, "/api/runtime/state").await,
        "stop response must be the latest immutable runtime snapshot"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn opposing_commands_return_the_snapshot_bound_to_each_completion() {
    let app = app();
    let start_request = send(
        app.clone(),
        Request::post("/api/runtime/start")
            .body(Body::empty())
            .expect("request"),
    );
    let stop_request = send(
        app.clone(),
        Request::post("/api/runtime/stop")
            .body(Body::empty())
            .expect("request"),
    );
    let (started, stopped) = tokio::join!(biased; start_request, stop_request);

    let started_body = response_json(started).await;
    let stopped_body = response_json(stopped).await;
    assert_eq!(started_body["running"], true);
    assert_eq!(started_body["epoch"], 1);
    assert_eq!(stopped_body["running"], false);
    assert_eq!(stopped_body["pipeline"]["phase"], "stopped");
    assert_eq!(stopped_body["pipeline"]["epoch"], 1);

    let prestarted = send(
        app.clone(),
        Request::post("/api/runtime/start")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(prestarted.status(), StatusCode::OK);

    let stop_request = send(
        app.clone(),
        Request::post("/api/runtime/stop")
            .body(Body::empty())
            .expect("request"),
    );
    let start_request = send(
        app.clone(),
        Request::post("/api/runtime/start")
            .body(Body::empty())
            .expect("request"),
    );
    let (stopped, restarted) = tokio::join!(biased; stop_request, start_request);

    let stopped_body = response_json(stopped).await;
    let restarted_body = response_json(restarted).await;
    assert_eq!(stopped_body["running"], false);
    assert_eq!(stopped_body["pipeline"]["phase"], "stopped");
    assert_eq!(stopped_body["pipeline"]["epoch"], 2);
    assert_eq!(restarted_body["running"], true);
    assert_eq!(restarted_body["epoch"], 3);
    assert_eq!(get_json(app, "/api/runtime/state").await["running"], true);
}

#[tokio::test(flavor = "current_thread")]
async fn concurrent_start_conflict_preserves_typed_http_error_semantics() {
    let app = app();
    let barrier = std::sync::Arc::new(Barrier::new(3));

    let first_app = app.clone();
    let first_barrier = barrier.clone();
    let first = tokio::spawn(async move {
        first_barrier.wait().await;
        send(
            first_app,
            Request::post("/api/runtime/start")
                .body(Body::empty())
                .expect("request"),
        )
        .await
    });

    let second_barrier = barrier.clone();
    let second = tokio::spawn(async move {
        second_barrier.wait().await;
        send(
            app,
            Request::post("/api/runtime/start")
                .body(Body::empty())
                .expect("request"),
        )
        .await
    });

    barrier.wait().await;
    let responses = [
        first.await.expect("first task"),
        second.await.expect("second task"),
    ];
    let success_count = responses
        .iter()
        .filter(|response| response.status() == StatusCode::OK)
        .count();
    let conflict_count = responses
        .iter()
        .filter(|response| response.status() == StatusCode::CONFLICT)
        .count();
    assert_eq!(success_count, 1);
    assert_eq!(conflict_count, 1);

    let conflict = responses
        .into_iter()
        .find(|response| response.status() == StatusCode::CONFLICT)
        .expect("conflict response");
    assert_eq!(
        response_json(conflict).await,
        json!({
            "code": "runtime_command_conflict",
            "detail": {
                "message": "runtime command start conflicts with an in-flight command"
            }
        })
    );
}

#[tokio::test]
async fn unknown_routes_and_methods_keep_axum_defaults() {
    let missing = send(
        app(),
        Request::get("/api/not-a-route")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    assert!(
        to_bytes(missing.into_body(), usize::MAX)
            .await
            .expect("body")
            .is_empty()
    );

    let wrong_method = send(
        app(),
        Request::get("/api/runtime/start")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(wrong_method.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        wrong_method.headers().get(ALLOW).expect("Allow header"),
        "POST"
    );
    assert!(
        to_bytes(wrong_method.into_body(), usize::MAX)
            .await
            .expect("body")
            .is_empty()
    );
}
