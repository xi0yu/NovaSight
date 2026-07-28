use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use novasight_api::{
    build_control_router, build_control_router_with_capabilities,
    build_control_router_with_platform_queries, build_control_router_with_services,
    build_control_router_with_shutdown,
};
use novasight_core::controller::recoil::RecoilConfig;
use novasight_core::{
    CaptureCapabilities, CaptureCapability, CaptureCapabilityProbe, CaptureProbeError, Clock,
    Detection, DetectionBatch, FrameStamp, MonotonicNanos, PointerDevice, RecordingPointerDevice,
    UncommissionedPointerDevice,
};
use novasight_pipeline::PipelineConfig;
use novasight_runtime::{
    ConfigService, PipelineState, RuntimeDependencies, RuntimeHandle, RuntimeSupervisor,
};
use novasight_store::config::YamlConfigRepository;
use serde_json::{Value, json};
use tower::ServiceExt;

static NEXT_CONFIG_DIRECTORY: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct StaticCaptureProbe;

impl CaptureCapabilityProbe for StaticCaptureProbe {
    fn probe(&self, device: &str) -> Result<CaptureCapabilities, CaptureProbeError> {
        Ok(CaptureCapabilities {
            available: true,
            device: device.to_owned(),
            capabilities: vec![CaptureCapability {
                pixel_format: "MJPG".to_owned(),
                width: 1_920,
                height: 1_080,
                fps_list: vec![120, 60],
            }],
            reason: String::new(),
        })
    }
}

struct ConfigDirectory(PathBuf);

impl ConfigDirectory {
    fn new() -> Self {
        let unique = NEXT_CONFIG_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "novasight-control-api-config-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for ConfigDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn commissioned_config(revision: u64, output_enabled: bool) -> String {
    format!(
        "revision: {revision}\ncontrol:\n  output_enabled: {output_enabled}\nhardware:\n  auto_connect: true\n  backend: native_udp\n  host: 127.0.0.1\n  port: 8888\n  uuid: A1B2C3D4\n  monitor_port: 5001\n  connect_timeout_ms: 3000\n  send_timeout_ms: 25\n  monitor_timeout_ms: 250\n  trigger_poll_interval_ms: 4\n"
    )
}

async fn request(runtime: &RuntimeHandle, method: &str, path: &str) -> (StatusCode, Value) {
    let app = build_control_router(runtime.clone());
    let request = Request::builder()
        .method(method)
        .uri(path)
        .body(Body::empty())
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    let body = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).expect("JSON body")
    };
    (status, body)
}

async fn shutdown(supervisor: RuntimeSupervisor, runtime: &RuntimeHandle) {
    runtime.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("join supervisor");
}

#[tokio::test]
async fn kmnet_buttons_projects_daemon_owned_trigger_cache() {
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    runtime.start().await.expect("start pipeline");
    runtime
        .set_trigger_active(true)
        .await
        .expect("set daemon trigger cache");

    let (status, body) = request(&runtime, "GET", "/api/executors/kmnet/buttons").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["available"], true);
    assert_eq!(body["left"], true);
    assert_eq!(body["right"], false);
    assert_eq!(body["managed_by_runtime"], true);

    runtime.stop().await.expect("stop pipeline");
    let (status, body) = request(&runtime, "GET", "/api/executors/kmnet/buttons").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["available"], false);
    assert_eq!(body["left"], false);
    assert_eq!(body["right"], false);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn versioned_config_api_preserves_pending_restart_while_hot_applying_output_gate() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(
        &path,
        "revision: 2\nserver:\n  port: 5174\nfuture:\n  keep: true\n",
    )
    .unwrap();
    let initial = YamlConfigRepository::load(&path).unwrap();
    let config = ConfigService::new(&path, initial);
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_services(runtime.clone(), Some(config), None);
    let request = Request::builder()
        .method("PATCH")
        .uri("/api/v1/config")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"section":"server","key":"port","value":6000,"expected_revision":2}"#,
        ))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["config"]["revision"], 3);
    assert_eq!(body["config"]["server"]["port"], 6000);
    assert_eq!(body["restart_required"], true);
    assert_eq!(body["applied"], false);
    assert_eq!(body["rolled_back"], false);
    let persisted: Value = serde_yaml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(persisted["future"]["keep"].as_bool(), Some(true));

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/api/v1/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"section":"control","key":"output_enabled","value":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let update: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(update["config"]["revision"], 4);
    assert_eq!(update["config"]["control"]["output_enabled"], false);
    assert_eq!(update["restart_required"], true);
    assert_eq!(update["applied"], true);
    assert_eq!(update["rolled_back"], false);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["revision"], 4);
    assert_eq!(body["server"]["port"], 6000);
    assert_eq!(body["control"]["output_enabled"], false);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn runtime_start_loads_pending_visual_pipeline_configuration_into_the_next_epoch() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    YamlConfigRepository::initialize_default(&path).unwrap();
    let initial = YamlConfigRepository::load(&path).unwrap();
    let config = ConfigService::new(&path, initial);
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_platform_queries(
        runtime.clone(),
        config.clone(),
        None,
        None,
        Arc::new(StaticCaptureProbe) as Arc<dyn CaptureCapabilityProbe>,
        false,
        None,
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/api/v1/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"section":"pipeline","key":"near_threshold_px","value":18.0}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(config.effective_revision(), 0);
    assert_eq!(config.snapshot().await.revision, 1);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/capture/select")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"device":"/dev/video0","preference":"manual","pixel_format":"MJPG","width":1920,"height":1080,"fps":120}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(config.effective_revision(), 2);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/runtime/start")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(config.effective_revision(), 2);
    assert_eq!(
        config
            .blocking_effective_snapshot()
            .pipeline
            .near_threshold_px,
        18.0
    );
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Running);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn runtime_start_keeps_process_owned_hardware_changes_behind_daemon_restart() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    YamlConfigRepository::initialize_default(&path).unwrap();
    let initial = YamlConfigRepository::load(&path).unwrap();
    let config = ConfigService::new(&path, initial);
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_services(runtime.clone(), Some(config.clone()), None);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/api/v1/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"section":"hardware","key":"host","value":"192.168.2.199"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/runtime/start")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["code"], "CONFIG_RESTART_REQUIRED");
    assert!(body["message"].as_str().unwrap().contains("hardware"));
    assert_eq!(config.effective_revision(), 0);
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn output_gate_config_is_persisted_and_applied_without_runtime_restart() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(&path, commissioned_config(0, true)).unwrap();
    let initial = YamlConfigRepository::load(&path).unwrap();
    let output_enabled = initial.control.output_enabled;
    let config = ConfigService::new(&path, initial);
    let (supervisor, runtime) = RuntimeSupervisor::spawn(
        RuntimeDependencies::recording().with_output_enabled(output_enabled),
    );
    runtime.start().await.unwrap();
    assert!(runtime.snapshot().pipeline_metrics.output_gate_open);
    let app = build_control_router_with_services(runtime.clone(), Some(config), None);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/api/v1/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"section":"control","key":"output_enabled","value":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["applied"], true);
    assert_eq!(body["restart_required"], false);
    assert_eq!(body["config"]["control"]["output_enabled"], false);
    assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"section":"control","key":"output_enabled","value":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(runtime.snapshot().pipeline_metrics.output_gate_open);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn trigger_mode_config_is_applied_to_the_running_pipeline_without_restart() {
    #[derive(Debug)]
    struct FixedClock;

    impl Clock for FixedClock {
        fn now(&self) -> MonotonicNanos {
            MonotonicNanos(1_008_000_000)
        }
    }

    fn batch(epoch: novasight_core::RuntimeEpoch, generation: u64) -> DetectionBatch {
        DetectionBatch::new(
            FrameStamp::new(epoch, generation, 1_000_000_000 + generation),
            640,
            640,
            vec![
                Detection::new(generation, 0, 380.0, 330.0, 40.0, 40.0, 0.95)
                    .expect("valid detection"),
            ],
        )
        .expect("valid batch")
    }

    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(
        &path,
        commissioned_config(0, true).replace(
            "control:\n  output_enabled: true",
            "control:\n  output_enabled: true\n  trigger_mode: always",
        ),
    )
    .unwrap();
    let initial = YamlConfigRepository::load(&path).unwrap();
    let config = ConfigService::new(&path, initial);
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let dependencies =
        RuntimeDependencies::new(Arc::new(FixedClock), pointer, PipelineConfig::default())
            .with_output_enabled(true);
    let (supervisor, runtime) = RuntimeSupervisor::spawn(dependencies);
    let started = runtime.start().await.unwrap();
    let app = build_control_router_with_services(runtime.clone(), Some(config.clone()), None);

    let response = app
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/api/v1/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"section":"control","key":"trigger_mode","value":"hardware"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["config"]["control"]["trigger_mode"], "hardware");
    assert_eq!(body["applied"], true);
    assert_eq!(body["restart_required"], false);
    assert_eq!(config.effective_revision(), 1);

    let epoch = started.pipeline.epoch.unwrap();
    runtime.submit_detection_batch(batch(epoch, 1)).unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(device.receipts().is_empty());

    let restarted = runtime.restart().await.unwrap();
    let epoch = restarted.pipeline.epoch.unwrap();
    runtime.submit_detection_batch(batch(epoch, 2)).unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(device.receipts().is_empty());

    runtime.set_trigger_active(true).await.unwrap();
    runtime.submit_detection_batch(batch(epoch, 3)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().is_empty() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(device.receipts().len(), 1);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn recoil_config_is_applied_to_the_running_pipeline_without_restart() {
    #[derive(Debug)]
    struct ManualClock(AtomicU64);

    impl Clock for ManualClock {
        fn now(&self) -> MonotonicNanos {
            MonotonicNanos(self.0.load(Ordering::Acquire))
        }
    }

    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(&path, commissioned_config(0, true)).unwrap();
    let initial = YamlConfigRepository::load(&path).unwrap();
    let config = ConfigService::new(&path, initial);
    let device = Arc::new(RecordingPointerDevice::default());
    let pointer: Arc<dyn PointerDevice> = device.clone();
    let clock = Arc::new(ManualClock(AtomicU64::new(1_008_000_000)));
    let runtime_clock: Arc<dyn Clock> = clock.clone();
    let dependencies = RuntimeDependencies::new(runtime_clock, pointer, PipelineConfig::default())
        .with_output_enabled(true);
    let (supervisor, runtime) = RuntimeSupervisor::spawn(dependencies);
    let started = runtime.start().await.unwrap();
    let app = build_control_router_with_services(runtime.clone(), Some(config.clone()), None);

    let response = app
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/api/v1/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"section":"control","key":"recoil","value":{"enabled":true,"require_target":true,"interval_ms":4,"y_counts":2}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["config"]["control"]["recoil"]["interval_ms"], 4);
    assert_eq!(body["config"]["control"]["recoil"]["y_counts"], 2);
    assert_eq!(body["applied"], true);
    assert_eq!(body["restart_required"], false);
    assert_eq!(config.effective_revision(), 1);

    runtime.set_trigger_active(true).await.unwrap();
    let epoch = started.pipeline.epoch.unwrap();
    for (generation, capture_ts, now_ns) in [
        (1, 1_000_000_000, 1_008_000_000),
        (2, 1_070_000_000, 1_078_000_000),
    ] {
        clock.0.store(now_ns, Ordering::Release);
        runtime
            .submit_detection_batch(
                DetectionBatch::new(
                    FrameStamp::new(epoch, generation, capture_ts),
                    640,
                    640,
                    vec![Detection::new(1, 0, 310.0, 311.2, 40.0, 40.0, 0.95).unwrap()],
                )
                .unwrap(),
            )
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let deadline = Instant::now() + Duration::from_secs(1);
    while device.receipts().len() < 2 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(
        device.receipts().len(),
        2,
        "pipeline metrics: {:?}",
        runtime.snapshot().pipeline_metrics
    );
    assert_eq!(device.receipts()[1].delta_y_counts, 2);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn failed_output_gate_hot_apply_never_advances_the_effective_revision() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(&path, commissioned_config(0, true)).unwrap();
    let initial = YamlConfigRepository::load(&path).unwrap();
    let config = ConfigService::new(&path, initial);
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    runtime.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("join supervisor");
    let app = build_control_router_with_services(runtime, Some(config.clone()), None);

    let response = app
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/api/v1/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"section":"control","key":"output_enabled","value":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(config.snapshot().await.revision, 0);
    assert_eq!(config.effective_revision(), 0);
    assert!(config.ensure_effective().await.is_ok());
    assert_eq!(YamlConfigRepository::load(&path).unwrap().revision, 0);
}

#[tokio::test]
async fn output_gate_persistence_failure_keeps_the_live_runtime_fail_closed() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(&path, commissioned_config(0, false)).unwrap();
    let initial = YamlConfigRepository::load(&path).unwrap();
    let config = ConfigService::new(&path, initial);
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    runtime.start().await.unwrap();
    assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);

    // Simulate an out-of-process writer after the service snapshot. The actor
    // must detect the optimistic revision race before opening physical output.
    fs::write(&path, commissioned_config(1, false)).unwrap();
    let app = build_control_router_with_services(runtime.clone(), Some(config.clone()), None);
    let response = app
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/api/v1/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"section":"control","key":"output_enabled","value":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);
    assert_eq!(config.snapshot().await.revision, 0);
    assert_eq!(config.effective_revision(), 0);
    let persisted = YamlConfigRepository::load(&path).unwrap();
    assert_eq!(persisted.revision, 1);
    assert!(!persisted.control.output_enabled);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn failed_disable_persistence_leaves_output_closed_and_reports_runtime_divergence() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(&path, commissioned_config(0, true)).unwrap();
    let initial = YamlConfigRepository::load(&path).unwrap();
    let config = ConfigService::new(&path, initial);
    let (supervisor, runtime) =
        RuntimeSupervisor::spawn(RuntimeDependencies::recording().with_output_enabled(true));
    runtime.start().await.unwrap();
    assert!(runtime.snapshot().pipeline_metrics.output_gate_open);

    // Lose the persistence CAS to an external writer. A failed close must
    // still retire physical output and make the config/runtime split visible.
    fs::write(&path, commissioned_config(1, true)).unwrap();
    let app = build_control_router_with_services(runtime.clone(), Some(config.clone()), None);
    let response = app
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/api/v1/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"section":"control","key":"output_enabled","value":false}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["code"], "CONFIG_OUTPUT_GATE_DISABLED_NOT_PERSISTED");
    assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);
    assert_eq!(config.snapshot().await.revision, 0);
    assert_eq!(config.effective_revision(), 0);
    assert_eq!(
        config.ensure_effective().await.unwrap_err().code(),
        "CONFIG_RUNTIME_DIVERGED"
    );
    let persisted = YamlConfigRepository::load(&path).unwrap();
    assert_eq!(persisted.revision, 1);
    assert!(persisted.control.output_enabled);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn studio_config_alias_uses_the_same_service_and_revision_guard() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(&path, "revision: 0\nreplay:\n  output_gate_open: false\n").unwrap();
    let initial = YamlConfigRepository::load(&path).unwrap();
    let config = ConfigService::new(&path, initial);
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_services(runtime.clone(), Some(config), None);
    let request = Request::builder()
        .method("POST")
        .uri("/api/config")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"section":"replay","key":"output_gate_open","value":true}"#,
        ))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["config"]["revision"], 1);
    assert_eq!(body["config"]["replay"]["output_gate_open"], true);
    assert_eq!(body["applied"], false);

    let request = Request::builder()
        .method("POST")
        .uri("/api/config")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"revision":1,"replay":{"output_gate_open":true,"frame_interval_ms":24}}"#,
        ))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["config"]["revision"], 2);
    assert_eq!(body["config"]["replay"]["frame_interval_ms"], 24);
    assert_eq!(body["restart_required"], true);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/runtime/start")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let error: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(error["code"], "CONFIG_RESTART_REQUIRED");
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/runtime/start")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let error: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(error["code"], "CONFIG_RESTART_REQUIRED");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/runtime/state")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let state: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(state["config"]["version"], 2);
    assert_eq!(state["config"]["effective_version"], 0);
    assert_eq!(state["config"]["restart_required"], true);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn whole_document_cannot_change_the_hot_output_gate() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(&path, "revision: 0\ncontrol:\n  output_enabled: false\n").unwrap();
    let before = fs::read(&path).unwrap();
    let config = ConfigService::new(&path, YamlConfigRepository::load(&path).unwrap());
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_services(runtime.clone(), Some(config.clone()), None);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"revision":0,"control":{"output_enabled":true}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(error["code"], "CONFIG_HOT_UPDATE_TRANSACTION_REQUIRED");
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(config.snapshot().await.revision, 0);
    assert_eq!(config.effective_revision(), 0);
    assert!(!runtime.snapshot().pipeline_metrics.output_gate_open);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn whole_document_cannot_remove_hardware_while_output_is_enabled() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(&path, commissioned_config(0, true)).unwrap();
    let before = fs::read(&path).unwrap();
    let config = ConfigService::new(&path, YamlConfigRepository::load(&path).unwrap());
    let (supervisor, runtime) =
        RuntimeSupervisor::spawn(RuntimeDependencies::recording().with_output_enabled(true));
    runtime.start().await.unwrap();
    assert!(runtime.snapshot().pipeline_metrics.output_gate_open);
    let app = build_control_router_with_services(runtime.clone(), Some(config.clone()), None);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/config")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"revision":0,"hardware":null}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(error["code"], "CONFIG_VALIDATION_ERROR");
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(config.snapshot().await.revision, 0);
    assert_eq!(config.effective_revision(), 0);
    assert!(runtime.snapshot().pipeline_metrics.output_gate_open);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn studio_config_schema_is_the_live_rust_config_contract() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(
        &path,
        r#"schema_version: 1
revision: 4
server:
  port: 6000
inference:
  enabled: true
  backend: deepstream_nvinfer
  device: cuda
  require_gpu: true
  allow_cpu_fallback: false
  confidence_threshold: 0.25
  nms_threshold: 0.45
  inference_input_deadline_ms: 55.0
  deepstream_parser_library: auto
  deepstream_io_mode: 2
  deepstream_batched_push_timeout_us: 0
  deepstream_component_id: 1
  deepstream_source_id: 0
  deepstream_probe_element: primary-infer
  deepstream_probe_pad: src
  deepstream_nvinfer_config: ""
  model_width: 640
  model_height: 640
  deepstream_startup_timeout_ms: 10000
  deepstream_shutdown_timeout_ms: 5000
  input_source: source.default
"#,
    )
    .unwrap();
    let config = ConfigService::new(&path, YamlConfigRepository::load(&path).unwrap());
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_services(runtime.clone(), Some(config), None);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/config/schema")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let schema: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(schema["version"], 8);
    assert_eq!(schema["values"]["revision"], 4);
    assert_eq!(schema["values"]["server"]["port"], 6000);
    assert_eq!(
        schema["values"]["inference"]["backend"],
        "deepstream_nvinfer"
    );
    assert_eq!(schema["values"]["inference"]["confidence_threshold"], 0.25);
    assert_eq!(schema["values"]["inference"]["nms_threshold"], 0.45);
    let inference = schema["sections"]
        .as_array()
        .unwrap()
        .iter()
        .find(|section| section["id"] == "inference")
        .unwrap();
    let backend = inference["fields"]
        .as_array()
        .unwrap()
        .iter()
        .find(|field| field["path"] == "inference.backend")
        .unwrap();
    assert_eq!(
        backend["options"],
        serde_json::json!(["deepstream_nvinfer"])
    );
    assert_eq!(backend["restart_required"], true);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn capture_state_projects_the_configured_device_and_supervisor_state() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(
        &path,
        "revision: 3\ncapture:\n  device: /dev/video7\n  backend: deepstream_nvinfer\n",
    )
    .unwrap();
    let config = ConfigService::new(&path, YamlConfigRepository::load(&path).unwrap());
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_services(runtime.clone(), Some(config), None);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/capture/state")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let capture: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(capture["device"], "/dev/video7");
    assert_eq!(capture["backend"], "deepstream_nvinfer");
    assert_eq!(capture["running"], false);
    assert_eq!(capture["state"], "stopped");
    assert_eq!(capture["available"], true);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn capture_capabilities_use_the_attached_platform_probe_for_get_and_post() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(
        &path,
        "revision: 3\ncapture:\n  device: /dev/video7\n  backend: deepstream_nvinfer\n",
    )
    .unwrap();
    let config = ConfigService::new(&path, YamlConfigRepository::load(&path).unwrap());
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_platform_queries(
        runtime.clone(),
        config,
        None,
        None,
        Arc::new(StaticCaptureProbe) as Arc<dyn CaptureCapabilityProbe>,
        false,
        None,
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/capture/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let configured: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(configured["device"], "/dev/video7");
    assert_eq!(configured["capabilities"][0]["pixel_format"], "MJPG");
    assert_eq!(configured["capabilities"][0]["fps_list"], json!([120, 60]));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/capture/capabilities")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"device":"/dev/video3"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let requested: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(requested["device"], "/dev/video3");

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn capture_selection_persists_a_concrete_profile_and_remains_startable() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(
        &path,
        r#"revision: 3
capture:
  device: /dev/video7
  backend: deepstream_nvinfer
  memory: nvmm
  preference: manual
  latest_only: true
  appsink_max_buffers: 1
  queue_leaky: downstream
  width: 1920
  height: 1080
  fps: 60
  pixel_format: NV12
  roi_left: 0
  roi_top: 0
  roi_width: 640
  roi_height: 640
"#,
    )
    .unwrap();
    let config = ConfigService::new(&path, YamlConfigRepository::load(&path).unwrap());
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_platform_queries(
        runtime.clone(),
        config,
        None,
        None,
        Arc::new(StaticCaptureProbe) as Arc<dyn CaptureCapabilityProbe>,
        false,
        None,
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/capture/select")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"device":"/dev/video3","preference":"auto_high_fps"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let selected: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(selected["available"], true);
    assert_eq!(selected["device"], "/dev/video3");
    assert_eq!(selected["profile"]["pixel_format"], "MJPG");
    assert_eq!(selected["profile"]["fps"], 120);
    assert_eq!(selected["profile"]["preference"], "manual");

    let persisted = YamlConfigRepository::load(&path).unwrap();
    let capture = persisted.capture.unwrap();
    assert_eq!(persisted.revision, 4);
    assert_eq!(capture.device, PathBuf::from("/dev/video3"));
    assert_eq!(capture.pixel_format, "MJPG");
    assert_eq!(capture.fps, 120);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/runtime/start")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/capture/select")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"device":"/dev/video3","preference":"auto_low_latency"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert_eq!(YamlConfigRepository::load(&path).unwrap().revision, 4);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/capture/stop")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let stopped: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(stopped["running"], false);
    assert_eq!(stopped["state"], "stopped");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/capture/select")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"device":"/dev/video3","preference":"auto_low_latency"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(YamlConfigRepository::load(&path).unwrap().revision, 5);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn studio_lifecycle_aliases_project_the_real_supervisor_and_config() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(
        &path,
        "revision: 5\nreplay:\n  enabled: true\ncapture:\n  device: /dev/video9\n  backend: deepstream_nvinfer\n",
    )
    .unwrap();
    let config = ConfigService::new(&path, YamlConfigRepository::load(&path).unwrap());
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_services(runtime.clone(), Some(config), None);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let health: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(health["ok"], true);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/runtime/start")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let started: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(started["running"], true);
    assert_eq!(started["accepted"], true);
    assert_eq!(started["epoch"], 1);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/runtime/state")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let state: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(state["running"], true);
    assert_eq!(state["pipeline"]["state"], "running");
    assert_eq!(state["pipeline"]["epoch"], 1);
    assert_eq!(state["config"]["version"], 5);
    assert_eq!(state["config"]["effective_version"], 5);
    assert_eq!(state["config"]["restart_required"], false);
    assert_eq!(state["capture"]["device"], "/dev/video9");
    assert_eq!(state["capture"]["backend"], "deepstream_nvinfer");
    assert_eq!(state["statistics"]["nvinfer_input_counter"], 0);
    assert_eq!(
        state["statistics"]["nvinfer_input_counter"],
        serde_json::to_value(runtime.snapshot().perception_metrics.input_buffers).unwrap()
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/runtime/stop")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let stopped: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(stopped["running"], false);
    assert_eq!(stopped["pipeline"]["state"], "stopped");

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn health_fails_closed_after_the_runtime_supervisor_exits() {
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router(runtime.clone());

    runtime
        .shutdown_daemon()
        .await
        .expect("shutdown runtime supervisor");
    supervisor.join().await.expect("join runtime supervisor");

    let response = app
        .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
        .await
        .expect("health response");
    assert_eq!(response.status(), StatusCode::OK);
    let health: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("health body"),
    )
    .expect("health JSON");
    assert_eq!(health["ok"], false);
}

#[tokio::test]
async fn studio_status_websocket_streams_real_supervisor_changes() {
    use futures_util::StreamExt;

    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(&path, "revision: 8\nreplay:\n  enabled: true\n").unwrap();
    let config = ConfigService::new(&path, YamlConfigRepository::load(&path).unwrap());
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let address = listener.local_addr().unwrap();
    let app = build_control_router_with_services(runtime.clone(), Some(config), None);
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve control API");
    });
    let (mut socket, _) =
        tokio_tungstenite::connect_async(format!("ws://{address}/ws/status?topic=summary"))
            .await
            .expect("connect Studio status socket");

    let initial = socket.next().await.unwrap().unwrap();
    let initial: Value = serde_json::from_str(initial.to_text().unwrap()).unwrap();
    assert_eq!(initial["kind"], "runtime_snapshot");
    assert_eq!(initial["topic"], "summary");
    assert_eq!(initial["full"], true);
    assert_eq!(initial["state"]["running"], false);
    assert_eq!(initial["state"]["config"]["version"], 8);

    runtime.start().await.unwrap();
    let running = loop {
        let frame = socket.next().await.unwrap().unwrap();
        let frame: Value = serde_json::from_str(frame.to_text().unwrap()).unwrap();
        if frame["state"]["running"] == true {
            break frame;
        }
    };
    assert_eq!(running["state"]["pipeline"]["epoch"], 1);
    assert_eq!(running["state"]["pipeline"]["state"], "running");

    socket.close(None).await.unwrap();
    server.abort();
    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn kmnet_diagnostics_are_real_supervisor_commands_and_never_dry_run_claims() {
    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(&path, commissioned_config(0, false)).unwrap();
    let config = ConfigService::new(&path, YamlConfigRepository::load(&path).unwrap());
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let dry_run =
        build_control_router_with_capabilities(runtime.clone(), Some(config.clone()), false, None);
    let request = || {
        Request::builder()
            .method("POST")
            .uri("/api/executors/kmnet/diagnostic-move")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"dx":4,"dy":-2}"#))
            .unwrap()
    };

    let response = dry_run.oneshot(request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let error: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(error["code"], "HARDWARE_OUTPUT_DISABLED");

    let production =
        build_control_router_with_capabilities(runtime.clone(), Some(config), true, None);
    let response = production.clone().oneshot(request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let error: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(error["code"], "output_gate_closed");

    let response = production
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/api/v1/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"section":"control","key":"output_enabled","value":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response = production.clone().oneshot(request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let result: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(result["sent"], true);
    assert_eq!(result["queued"], false);
    assert_eq!(result["steps_sent"], 1);
    assert_eq!(result["receipt"]["epoch"], 0);
    assert_eq!(result["receipt"]["generation"], 1);
    assert_eq!(result["receipt"]["delta_x_counts"], 4);
    assert_eq!(result["status"]["managed_by_runtime"], true);

    let response = production
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/executors")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let executors: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(executors["executors"]["kmnet"]["connected"], true);
    assert_eq!(executors["executors"]["kmnet"]["diagnostic_move_count"], 1);
    assert_eq!(executors["executors"]["kmnet"]["last_diagnostic_dx"], 4);
    assert_eq!(executors["executors"]["kmnet"]["last_diagnostic_dy"], -2);
    assert_eq!(executors["executors"]["kmnet"]["managed_by_runtime"], true);

    let response = production
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/executors/kmnet/connect")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let error: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(error["code"], "pipeline_unavailable");

    runtime.start().await.unwrap();
    let response = production.oneshot(request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Running);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn uncommissioned_diagnostic_returns_a_stable_conflict() {
    #[derive(Debug)]
    struct FixedClock;

    impl Clock for FixedClock {
        fn now(&self) -> MonotonicNanos {
            MonotonicNanos(1)
        }
    }

    let directory = ConfigDirectory::new();
    let path = directory.0.join("novasight.yaml");
    fs::write(
        &path,
        "revision: 0\ncontrol:\n  output_enabled: false\nhardware: {}\n",
    )
    .unwrap();
    let config = ConfigService::new(&path, YamlConfigRepository::load(&path).unwrap());
    let clock: Arc<dyn Clock> = Arc::new(FixedClock);
    let device: Arc<dyn PointerDevice> = Arc::new(UncommissionedPointerDevice);
    let (supervisor, runtime) = RuntimeSupervisor::spawn(RuntimeDependencies::new(
        clock,
        device,
        PipelineConfig::default(),
    ));
    let app =
        build_control_router_with_capabilities(runtime.clone(), Some(config.clone()), true, None);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/runtime/state")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let status: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let kmnet = &status["executor"]["executors"]["kmnet"];
    assert_eq!(kmnet["connection_state"], "uncommissioned");
    assert_eq!(kmnet["retryable"], false);
    assert!(status["fatal_error"].is_null());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri("/api/v1/config")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"section":"control","key":"output_enabled","value":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let error: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(error["code"], "device_uncommissioned");
    assert_eq!(config.snapshot().await.revision, 0);
    let persisted = YamlConfigRepository::load(&path).unwrap();
    assert_eq!(persisted.revision, 0);
    assert!(!persisted.control.output_enabled);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/executors/kmnet/diagnostic-move")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"dx":1,"dy":0}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CONFLICT);
    let error: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(error["code"], "device_uncommissioned");
    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn status_is_the_supervisors_current_immutable_snapshot() {
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();

    let (status, body) = request(&runtime, "GET", "/api/v1/status").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, serde_json::to_value(runtime.snapshot()).unwrap());

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn lifecycle_routes_delegate_to_the_single_runtime_handle() {
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();

    let (status, started) = request(&runtime, "POST", "/api/v1/runtime/start").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(started["pipeline"]["state"], "running");
    assert_eq!(started["pipeline"]["epoch"], 1);

    let (status, restarted) = request(&runtime, "POST", "/api/v1/runtime/restart").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(restarted["pipeline"]["state"], "running");
    assert_eq!(restarted["pipeline"]["epoch"], 2);

    let (status, stopped) = request(&runtime, "POST", "/api/v1/runtime/stop").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(stopped["pipeline"]["state"], "stopped");
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Stopped);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn emergency_stop_is_exposed_without_a_daemon_shutdown_route() {
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    request(&runtime, "POST", "/api/v1/runtime/start").await;

    let (status, stopped) = request(&runtime, "POST", "/api/v1/runtime/emergency-stop").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(stopped["pipeline"]["state"], "stopped");
    assert_eq!(stopped["subsystems"]["device"]["state"], "unavailable");

    let (status, _) = request(&runtime, "POST", "/api/v1/runtime/shutdown-daemon").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn closed_supervisor_returns_a_stable_service_unavailable_error() {
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    runtime.shutdown_daemon().await.expect("shutdown daemon");
    supervisor.join().await.expect("join supervisor");

    let (status, body) = request(&runtime, "POST", "/api/v1/runtime/start").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["code"], "supervisor_closed");
    assert!(
        body["message"]
            .as_str()
            .is_some_and(|message| !message.is_empty())
    );
}

#[tokio::test]
async fn internal_pipeline_start_failure_is_not_misreported_as_a_client_conflict() {
    #[derive(Debug)]
    struct FixedClock;

    impl Clock for FixedClock {
        fn now(&self) -> MonotonicNanos {
            MonotonicNanos(1_000_000_000)
        }
    }

    let clock: Arc<dyn Clock> = Arc::new(FixedClock);
    let device: Arc<dyn PointerDevice> = Arc::new(RecordingPointerDevice::default());
    let dependencies = RuntimeDependencies::new(
        clock,
        device,
        PipelineConfig {
            recoil: RecoilConfig {
                interval_ms: 0,
                ..RecoilConfig::default()
            },
            ..PipelineConfig::default()
        },
    );
    let (supervisor, runtime) = RuntimeSupervisor::spawn(dependencies);

    let (status, body) = request(&runtime, "POST", "/api/v1/runtime/start").await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body["code"], "pipeline_rejected");

    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn events_websocket_sends_initial_and_changed_supervisor_snapshots() {
    use futures_util::StreamExt;

    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let address = listener.local_addr().expect("server address");
    let app = build_control_router(runtime.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve control API");
    });
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{address}/api/v1/events"))
        .await
        .expect("connect events socket");

    let initial = socket.next().await.expect("initial frame").expect("frame");
    let initial: Value =
        serde_json::from_str(initial.to_text().expect("text frame")).expect("initial event JSON");
    assert_eq!(initial["kind"], "runtime_snapshot");
    assert_eq!(initial["snapshot"]["pipeline"]["state"], "stopped");

    runtime.start().await.expect("start runtime");
    let changed = loop {
        let changed = socket.next().await.expect("changed frame").expect("frame");
        let changed: Value = serde_json::from_str(changed.to_text().expect("text frame"))
            .expect("changed event JSON");
        if changed["snapshot"]["pipeline"]["state"] == "running" {
            break changed;
        }
    };
    assert_eq!(changed["snapshot"]["pipeline"]["state"], "running");
    assert_eq!(changed["snapshot"]["pipeline"]["epoch"], 1);

    socket.close(None).await.expect("close socket");
    server.abort();
    shutdown(supervisor, &runtime).await;
}

#[tokio::test]
async fn daemon_shutdown_closes_events_websocket_and_releases_axum_drain() {
    use futures_util::StreamExt;

    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let address = listener.local_addr().expect("server address");
    let app = build_control_router_with_shutdown(runtime.clone(), Some(shutdown_rx.clone()));
    let mut graceful_rx = shutdown_rx;
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                if !*graceful_rx.borrow() {
                    let _ = graceful_rx.changed().await;
                }
            })
            .await
            .expect("serve control API");
    });
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{address}/api/v1/events"))
        .await
        .expect("connect events socket");
    socket.next().await.expect("initial frame").expect("frame");

    shutdown_tx.send_replace(true);
    let closed = tokio::time::timeout(std::time::Duration::from_secs(1), socket.next())
        .await
        .expect("events socket closes on daemon shutdown");
    assert!(
        matches!(
            closed,
            None | Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_))) | Some(Err(_))
        ),
        "unexpected frame after shutdown: {closed:?}"
    );
    tokio::time::timeout(std::time::Duration::from_secs(1), server)
        .await
        .expect("Axum graceful drain completes")
        .expect("server task");

    shutdown(supervisor, &runtime).await;
}
