use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, to_bytes};
use novasight_api::build_control_router_with_control_plane;
use novasight_core::{Clock, MonotonicNanos, RuntimeEpoch};
use novasight_pipeline::{
    ModelCandidate, PerceptionAdapter, PerceptionError, PerceptionEvent, PerceptionModelContract,
    PerceptionSession, PipelineIngress,
};
use novasight_runtime::{OfflineModelJobRunner, RuntimeDependencies, RuntimeSupervisor};
use novasight_store::model_catalog::SqliteModelCatalog;
use rusqlite::Connection;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tower::ServiceExt;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "novasight-model-ingress-api-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[derive(Debug)]
struct TestClock;

impl Clock for TestClock {
    fn now(&self) -> MonotonicNanos {
        MonotonicNanos(1_000_000_000)
    }
}

#[derive(Debug)]
struct ReadyAdapter;

impl PerceptionAdapter for ReadyAdapter {
    fn preflight_model(
        &self,
        _candidate: &ModelCandidate,
    ) -> Result<Option<PerceptionModelContract>, PerceptionError> {
        Ok(None)
    }

    fn start(
        &self,
        _epoch: RuntimeEpoch,
        _ingress: PipelineIngress,
        _clock: Arc<dyn Clock>,
        _events: std::sync::mpsc::SyncSender<PerceptionEvent>,
    ) -> Result<Box<dyn PerceptionSession>, PerceptionError> {
        Ok(Box::new(ReadySession))
    }
}

#[derive(Debug)]
struct ReadySession;

impl PerceptionSession for ReadySession {
    fn shutdown(&mut self) -> Result<(), PerceptionError> {
        Ok(())
    }
}

#[tokio::test]
async fn rust_control_plane_owns_inspect_configure_probe_and_publish_sequence() {
    let directory = TestDirectory::new();
    let (catalog, engine_path, checksum) = seed_catalog(&directory.0);
    let helper = write_success_helper(&directory.0, &engine_path, &checksum);
    let runner = OfflineModelJobRunner::new(
        "/bin/sh",
        helper,
        &directory.0,
        directory.0.join("libnvdsinfer_custom_impl.so"),
        Duration::from_secs(2),
        1024 * 1024,
    )
    .unwrap();
    let dependencies = RuntimeDependencies::new(
        Arc::new(TestClock),
        Arc::new(novasight_core::RecordingPointerDevice::default()),
        novasight_pipeline::PipelineConfig::default(),
    )
    .with_perception(Arc::new(ReadyAdapter))
    .with_model_catalog(catalog.clone())
    .with_model_jobs(runner);
    let (supervisor, runtime) = RuntimeSupervisor::spawn(dependencies);
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );

    let inspected =
        request_json(app.clone(), "POST", "/api/models/artifacts/1/inspect", None).await;
    assert_eq!(inspected["profile"]["status"], "NEEDS_CONFIGURATION");
    assert_eq!(
        catalog.runtime_artifact_by_id(1).unwrap().artifact.status,
        "pending"
    );
    assert_eq!(
        catalog
            .runtime_artifact_by_id(1)
            .unwrap()
            .version
            .input_shape,
        "1x3x320x320"
    );

    std::fs::write(&engine_path, b"engine-replaced").unwrap();
    let stale = request_json(app.clone(), "GET", "/api/models/artifacts/1/profile", None).await;
    assert_eq!(stale["profile"]["status"], "UNINSPECTED");
    assert_eq!(
        stale["profile"]["validation"]["issues"],
        json!(["ENGINE_CONTENT_CHANGED"])
    );
    std::fs::write(&engine_path, b"real-engine-bytes").unwrap();

    let configured = request_json(
        app.clone(),
        "PUT",
        "/api/models/artifacts/1/profile",
        Some(json!({
            "color_format": "RGB",
            "scale": 0.00392156862745098,
            "resize_mode": "direct",
            "parser_type": "yolov8_raw",
            "class_count": 1,
            "labels": ["target"],
            "bbox_format": "xywh",
            "has_objectness": false
        })),
    )
    .await;
    assert_eq!(configured["profile"]["status"], "READY_FOR_PROBE");
    assert_eq!(
        catalog.runtime_artifact_by_id(1).unwrap().version.classes,
        ["target"]
    );

    let stored = request_json(app.clone(), "GET", "/api/models/artifacts/1/profile", None).await;
    assert_eq!(stored["profile"]["decoder"]["parser_type"], "yolov8_raw");

    let recommendation = request_json(
        app.clone(),
        "GET",
        "/api/models/artifacts/1/deepstream/recommendation",
        None,
    )
    .await;
    assert_eq!(recommendation["artifact_id"], 1);
    assert_eq!(recommendation["artifact_path"], "model.engine");
    assert_eq!(recommendation["recommendation"]["input_name"], "images");
    assert_eq!(
        recommendation["recommendation"]["input_shape"],
        json!([1, 3, 320, 320])
    );
    assert_eq!(recommendation["recommendation"]["output_name"], "output0");
    assert_eq!(recommendation["recommendation"]["class_count"], 1);
    assert_eq!(
        recommendation["recommendation"]["runtime_precision"],
        "fp32"
    );
    assert_eq!(recommendation["class_names"], json!(["target"]));
    assert_eq!(recommendation["output_has_objectness"], false);
    assert_eq!(
        recommendation["sources"]["input_contract"],
        "model_profile_receipt"
    );
    assert_eq!(
        recommendation["io_tensors"][1],
        json!({"name":"output0","shape":[1,5,2100],"dtype":"float32","mode":"output"})
    );

    let probed = request_json(
        app.clone(),
        "POST",
        "/api/models/artifacts/1/probe",
        Some(json!({"input_mode": "fixed"})),
    )
    .await;
    assert_eq!(probed["profile"]["status"], "VALIDATED");
    assert_eq!(probed["report"]["detection_batch_ok"], true);
    assert_eq!(
        catalog.runtime_artifact_by_id(1).unwrap().artifact.status,
        "ready"
    );

    let latest = request(
        app.clone(),
        "POST",
        "/api/models/artifacts/1/probe",
        Some(json!({"input_mode": "latest"})),
    )
    .await;
    assert_eq!(latest.status(), axum::http::StatusCode::CONFLICT);
    let latest_error: Value =
        serde_json::from_slice(&to_bytes(latest.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(latest_error["code"], "MODEL_LATEST_FRAME_UNAVAILABLE");

    let published = request_json(
        app.clone(),
        "POST",
        "/api/models/projects/1/publish",
        Some(json!({"artifact_id": 1, "parser_preset": "yolov8"})),
    )
    .await;
    assert_eq!(published["deployment"]["artifact_id"], 1);

    let active_mutation =
        request(app.clone(), "POST", "/api/models/artifacts/1/inspect", None).await;
    assert_eq!(active_mutation.status(), axum::http::StatusCode::CONFLICT);
    let error: Value = serde_json::from_slice(
        &to_bytes(active_mutation.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(error["code"], "MODEL_ARTIFACT_ACTIVE");

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_cancels_a_running_model_worker_before_supervisor_acknowledges_stop() {
    let directory = TestDirectory::new();
    let (catalog, _engine_path, _checksum) = seed_catalog(&directory.0);
    let marker = directory.0.join("worker-entered");
    let helper = directory.0.join("blocking-worker.py");
    std::fs::write(
        &helper,
        format!(
            "from pathlib import Path\nimport subprocess, sys, time\nPath({marker:?}).write_text('entered')\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\ntime.sleep(30)\n",
            marker = marker
        ),
    )
    .unwrap();
    let runner = OfflineModelJobRunner::new(
        "/usr/bin/python3",
        helper,
        &directory.0,
        directory.0.join("parser.so"),
        Duration::from_secs(60),
        1024,
    )
    .unwrap();
    let dependencies = RuntimeDependencies::new(
        Arc::new(TestClock),
        Arc::new(novasight_core::RecordingPointerDevice::default()),
        novasight_pipeline::PipelineConfig::default(),
    )
    .with_perception(Arc::new(ReadyAdapter))
    .with_model_catalog(catalog.clone())
    .with_model_jobs(runner);
    let (supervisor, runtime) = RuntimeSupervisor::spawn(dependencies);
    runtime.start().await.unwrap();
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );

    let inspect =
        tokio::spawn(
            async move { request(app, "POST", "/api/models/artifacts/1/inspect", None).await },
        );
    tokio::time::timeout(Duration::from_secs(2), async {
        while !marker.is_file() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("worker entered");

    let stopped = tokio::time::timeout(Duration::from_secs(2), runtime.stop())
        .await
        .expect("stop must interrupt the helper")
        .unwrap();
    assert_eq!(
        stopped.pipeline.state,
        novasight_runtime::PipelineState::Stopped
    );
    let response = inspect.await.unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::CONFLICT);
    let error: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(error["code"], "MODEL_INGRESS_CANCELLED");
    assert_eq!(
        catalog.runtime_artifact_by_id(1).unwrap().artifact.status,
        "pending"
    );

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
}

#[tokio::test]
async fn model_worker_timeout_and_output_limit_are_enforced_by_rust() {
    let timeout_directory = TestDirectory::new();
    let (timeout_catalog, _, _) = seed_catalog(&timeout_directory.0);
    let timeout_helper = timeout_directory.0.join("timeout-worker.py");
    std::fs::write(&timeout_helper, "import time\ntime.sleep(30)\n").unwrap();
    let timeout_runner = OfflineModelJobRunner::new(
        "/usr/bin/python3",
        timeout_helper,
        &timeout_directory.0,
        timeout_directory.0.join("parser.so"),
        Duration::from_millis(50),
        1024,
    )
    .unwrap();
    let (timeout_supervisor, timeout_runtime) = RuntimeSupervisor::spawn(
        RuntimeDependencies::recording()
            .with_model_catalog(timeout_catalog.clone())
            .with_model_jobs(timeout_runner),
    );
    let timeout_app = build_control_router_with_control_plane(
        timeout_runtime.clone(),
        None,
        None,
        timeout_catalog,
        false,
        None,
    );
    let timeout_response =
        request(timeout_app, "POST", "/api/models/artifacts/1/inspect", None).await;
    assert_eq!(
        timeout_response.status(),
        axum::http::StatusCode::GATEWAY_TIMEOUT
    );
    timeout_runtime.shutdown_daemon().await.unwrap();
    timeout_supervisor.join().await.unwrap();

    let output_directory = TestDirectory::new();
    let (output_catalog, _, _) = seed_catalog(&output_directory.0);
    let output_helper = output_directory.0.join("output-worker.py");
    std::fs::write(&output_helper, "print('x' * 4096)\n").unwrap();
    let output_runner = OfflineModelJobRunner::new(
        "/usr/bin/python3",
        output_helper,
        &output_directory.0,
        output_directory.0.join("parser.so"),
        Duration::from_secs(2),
        128,
    )
    .unwrap();
    let (output_supervisor, output_runtime) = RuntimeSupervisor::spawn(
        RuntimeDependencies::recording()
            .with_model_catalog(output_catalog.clone())
            .with_model_jobs(output_runner),
    );
    let output_app = build_control_router_with_control_plane(
        output_runtime.clone(),
        None,
        None,
        output_catalog,
        false,
        None,
    );
    let output_response =
        request(output_app, "POST", "/api/models/artifacts/1/inspect", None).await;
    assert_eq!(
        output_response.status(),
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    );
    let output_error: Value = serde_json::from_slice(
        &to_bytes(output_response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(output_error["code"], "MODEL_INGRESS_INTERNAL");
    output_runtime.shutdown_daemon().await.unwrap();
    output_supervisor.join().await.unwrap();
}

#[tokio::test]
async fn helper_replacement_after_daemon_composition_is_rejected() {
    let directory = TestDirectory::new();
    let (catalog, _, _) = seed_catalog(&directory.0);
    let helper = directory.0.join("identity-worker.py");
    std::fs::write(&helper, "print('{}')\n").unwrap();
    let runner = OfflineModelJobRunner::new(
        "/usr/bin/python3",
        &helper,
        &directory.0,
        directory.0.join("parser.so"),
        Duration::from_secs(2),
        1024,
    )
    .unwrap();
    std::fs::write(&helper, "print('{\"replaced\":true}')\n").unwrap();
    let (supervisor, runtime) = RuntimeSupervisor::spawn(
        RuntimeDependencies::recording()
            .with_model_catalog(catalog.clone())
            .with_model_jobs(runner),
    );
    let app =
        build_control_router_with_control_plane(runtime.clone(), None, None, catalog, false, None);

    let response = request(app, "POST", "/api/models/artifacts/1/inspect", None).await;
    assert_eq!(
        response.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    let error: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(error["code"], "MODEL_INGRESS_UNAVAILABLE");

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
}

#[tokio::test]
async fn failed_worker_validation_restores_the_previous_manifest() {
    let directory = TestDirectory::new();
    let (catalog, engine_path, _) = seed_catalog(&directory.0);
    let manifest = engine_path.with_file_name("model.engine.manifest.json");
    let previous = b"{\"preserved\":true}";
    std::fs::write(&manifest, previous).unwrap();
    let helper = directory.0.join("corrupting-worker.py");
    std::fs::write(
        &helper,
        format!(
            "from pathlib import Path\nPath({manifest:?}).write_text('{{\\\"corrupt\\\":true}}')\nprint('{{}}')\n"
        ),
    )
    .unwrap();
    let runner = OfflineModelJobRunner::new(
        "/usr/bin/python3",
        helper,
        &directory.0,
        directory.0.join("parser.so"),
        Duration::from_secs(2),
        1024,
    )
    .unwrap();
    let (supervisor, runtime) = RuntimeSupervisor::spawn(
        RuntimeDependencies::recording()
            .with_model_catalog(catalog.clone())
            .with_model_jobs(runner),
    );
    let app =
        build_control_router_with_control_plane(runtime.clone(), None, None, catalog, false, None);

    let response = request(app, "POST", "/api/models/artifacts/1/inspect", None).await;
    assert_eq!(
        response.status(),
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(std::fs::read(&manifest).unwrap(), previous);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
}

fn seed_catalog(root: &Path) -> (SqliteModelCatalog, PathBuf, String) {
    let database = root.join("novasight.db");
    let model_root = root.join("models");
    let engine_dir = model_root.join("detector/v1");
    std::fs::create_dir_all(&engine_dir).unwrap();
    let engine_path = engine_dir.join("model.engine");
    std::fs::write(&engine_path, b"real-engine-bytes").unwrap();
    let checksum = format!("{:x}", Sha256::digest(b"real-engine-bytes"));
    let catalog = SqliteModelCatalog::open_with_model_root(&database, &model_root).unwrap();
    let connection = Connection::open(database).unwrap();
    connection
        .execute(
            "INSERT INTO model_projects(id, name, description) VALUES (1, 'detector', 'primary')",
            [],
        )
        .unwrap();
    connection.execute("INSERT INTO model_versions(id, project_id, version, source_kind, source_path, classes_json, input_shape) VALUES (1, 1, 'v1', 'onnx', 'model.onnx', '[\"unknown\"]', 'engine-probe-required')", []).unwrap();
    connection.execute("INSERT INTO model_artifacts(id, version_id, kind, path, checksum, status) VALUES (1, 1, 'engine', 'model.engine', '', 'pending')", []).unwrap();
    (catalog, engine_path, checksum)
}

fn write_success_helper(root: &Path, engine_path: &Path, checksum: &str) -> PathBuf {
    let profile_path = engine_path.with_file_name("model.engine.manifest.json");
    let base_profile = |status: &str, validated: bool| {
        json!({
            "schema_version": 1,
            "model_id": format!("sha256:{checksum}"),
            "display_name": "detector",
            "status": status,
            "engine": {
                "path": engine_path.canonicalize().unwrap().to_string_lossy(),
                "sha256": format!("sha256:{checksum}"),
                "file_size": 17,
                "modified_at_ns": 1
            },
            "inspection": {"deserialize_ok": true, "compatible": true},
            "input": {"name": "images", "runtime_shape": [1,3,320,320], "engine_shape": [1,3,320,320], "dtype": "float32", "layout": "NCHW", "profile_index": 0},
            "outputs": [{"name": "output0", "shape": [1,5,2100], "engine_shape": [1,5,2100], "dtype": "float32"}],
            "preprocess": {"color_format": "RGB", "scale": 0.00392156862745098, "offsets": [], "mean": [], "std": [], "resize_mode": "direct", "symmetric_padding": false, "padding_value": 0.0},
            "decoder": {"parser_type": "yolov8_raw", "class_count": 1, "bbox_format": "xywh", "has_objectness": false},
            "postprocess": {"confidence_threshold": 0.25, "nms_threshold": 0.45, "max_detections": 300},
            "labels": ["target"],
            "parser_candidates": [],
            "validation": {
                "status": if validated { "validated" } else { "not_run" },
                "validated_at": "",
                "engine_execution_ok": validated,
                "decoder_ok": validated,
                "nms_ok": validated,
                "detection_batch_ok": validated,
                "profile_fingerprint": if validated { "sha256:test" } else { "" },
                "issues": []
            }
        })
    };
    let inspected = base_profile("NEEDS_CONFIGURATION", false);
    let configured = base_profile("READY_FOR_PROBE", false);
    let validated = base_profile("VALIDATED", true);
    let inspect_manifest =
        json!({"schema_version":1,"manifest_kind":"novasight_model","model_profile":inspected});
    let configure_manifest =
        json!({"schema_version":1,"manifest_kind":"novasight_model","model_profile":configured});
    let probe_manifest = json!({
        "schema_version": 1,
        "manifest_kind": "novasight_model",
        "model_id": format!("sha256:{checksum}"),
        "display_name": "detector",
        "artifact": {"engine_path":"model.engine","sha256":checksum,"size_bytes":17},
        "runtime": {"backend":"custom_tensorrt","precision":"fp32","batch_size":1},
        "input": {"name":"images","shape":[1,3,320,320],"dtype":"float32","layout":"NCHW","color_format":"RGB","scale_factor":0.00392156862745098,"maintain_aspect_ratio":false,"symmetric_padding":false},
        "output": {"name":"output0","shape":[1,5,2100],"dtype":"float32","layout":"NCHW","format":"yolo_cxcywh_class_scores","class_count":1,"class_names":["target"],"has_objectness":false,"scores_are_sigmoid":true,"coordinate_mode":"pixel"},
        "postprocess": {"parser":"yolo","parser_preset":"yolov8","confidence_threshold":0.25,"nms_iou_threshold":0.45,"class_aware_nms":true,"max_detections":300},
        "validated": true,
        "model_fingerprint": "",
        "model_profile": validated
    });
    let response = |profile: &Value, report: Option<Value>| {
        let mut value = json!({
            "profile_path": profile_path.canonicalize().unwrap_or_else(|_| profile_path.clone()).to_string_lossy(),
            "profile": profile
        });
        if let Some(report) = report {
            value["report"] = report;
        }
        value
    };
    let inspect_response = response(&inspect_manifest["model_profile"], None);
    let configure_response = response(&configure_manifest["model_profile"], None);
    let probe_response = response(
        &probe_manifest["model_profile"],
        Some(json!({
            "status":"validated",
            "engine_execution_ok":true,
            "output_tensor_ok":true,
            "decoder_ok":true,
            "nms_ok":true,
            "detection_batch_ok":true,
            "preprocess_ms":0.1,
            "inference_ms":1.0,
            "decode_ms":0.1,
            "nms_ms":0.0,
            "issues":[]
        })),
    );
    let shell = format!(
        "case \"$1\" in\ninspect) printf '%s' '{inspect_manifest}' > '{profile_path}'; printf '%s' '{inspect_response}' ;;\nconfigure) cat >/dev/null; printf '%s' '{configure_manifest}' > '{profile_path}'; printf '%s' '{configure_response}' ;;\nprobe) printf '%s' '{probe_manifest}' > '{profile_path}'; printf '%s' '{probe_response}' ;;\n*) exit 64 ;;\nesac\n",
        inspect_manifest = inspect_manifest,
        configure_manifest = configure_manifest,
        probe_manifest = probe_manifest,
        inspect_response = inspect_response,
        configure_response = configure_response,
        probe_response = probe_response,
        profile_path = profile_path.display(),
    );
    let helper = root.join("model-worker.sh");
    std::fs::write(&helper, shell).unwrap();
    helper
}

async fn request_json(app: Router, method: &str, path: &str, body: Option<Value>) -> Value {
    let response = request(app, method, path, body).await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
}

async fn request(
    app: Router,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> axum::response::Response {
    let mut builder = axum::http::Request::builder().method(method).uri(path);
    let body = match body {
        Some(body) => {
            builder = builder.header("content-type", "application/json");
            Body::from(serde_json::to_vec(&body).unwrap())
        }
        None => Body::empty(),
    };
    app.oneshot(builder.body(body).unwrap()).await.unwrap()
}
