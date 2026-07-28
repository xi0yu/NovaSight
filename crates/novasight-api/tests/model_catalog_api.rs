use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use novasight_api::build_control_router_with_control_plane;
use novasight_core::{Clock, MonotonicNanos, RecordingPointerDevice, RuntimeEpoch};
use novasight_pipeline::{
    ModelCandidate, PerceptionAdapter, PerceptionError, PerceptionEvent, PerceptionModelContract,
    PerceptionSession, PipelineConfig, PipelineIngress,
};
use novasight_runtime::{AppConfig, ConfigService, RuntimeDependencies, RuntimeSupervisor};
use novasight_store::model_catalog::SqliteModelCatalog;
use rusqlite::Connection;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tower::ServiceExt;

#[derive(Debug)]
struct RejectingModelAdapter;

#[derive(Debug)]
struct TestClock;

impl Clock for TestClock {
    fn now(&self) -> MonotonicNanos {
        MonotonicNanos(1_000_000_000)
    }
}

impl PerceptionAdapter for RejectingModelAdapter {
    fn preflight(&self) -> Result<(), PerceptionError> {
        Err(PerceptionError::new("candidate engine identity rejected"))
    }

    fn start(
        &self,
        _epoch: RuntimeEpoch,
        _ingress: PipelineIngress,
        _clock: Arc<dyn Clock>,
        _events: std::sync::mpsc::SyncSender<PerceptionEvent>,
    ) -> Result<Box<dyn PerceptionSession>, PerceptionError> {
        unreachable!("rejected candidate must not start")
    }
}

#[derive(Debug)]
struct CatalogAwareAdapter {
    catalog: SqliteModelCatalog,
    rejected_artifact_id: i64,
}

impl CatalogAwareAdapter {
    fn validate_artifact(&self, artifact_id: i64) -> Result<(), PerceptionError> {
        if artifact_id == self.rejected_artifact_id {
            return Err(PerceptionError::new(format!(
                "artifact {artifact_id} failed readiness"
            )));
        }
        Ok(())
    }

    fn validate_active(&self) -> Result<(), PerceptionError> {
        let active = self
            .catalog
            .active_model()
            .map_err(|error| PerceptionError::new(error.to_string()))?
            .ok_or_else(|| PerceptionError::new("model-not-loaded"))?;
        self.validate_artifact(active.artifact.id)
    }
}

impl PerceptionAdapter for CatalogAwareAdapter {
    fn preflight(&self) -> Result<(), PerceptionError> {
        self.validate_active()
    }

    fn preflight_model(
        &self,
        candidate: &ModelCandidate,
    ) -> Result<Option<PerceptionModelContract>, PerceptionError> {
        self.validate_artifact(candidate.artifact_id)?;
        Ok(None)
    }

    fn start(
        &self,
        _epoch: RuntimeEpoch,
        _ingress: PipelineIngress,
        _clock: Arc<dyn Clock>,
        _events: std::sync::mpsc::SyncSender<PerceptionEvent>,
    ) -> Result<Box<dyn PerceptionSession>, PerceptionError> {
        self.validate_active()?;
        Ok(Box::new(TestPerceptionSession))
    }
}

#[tokio::test]
async fn catalog_register_creates_an_idempotent_external_engine_reference() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-register-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    let model_root = directory.join("models");
    std::fs::create_dir_all(model_root.join("nested")).unwrap();
    let engine = model_root.join("nested/detector.engine");
    std::fs::write(&engine, b"opaque-engine").unwrap();
    let catalog =
        SqliteModelCatalog::open_with_model_root(directory.join("novasight.db"), &model_root)
            .unwrap();
    let (supervisor, runtime) = RuntimeSupervisor::spawn(
        RuntimeDependencies::recording().with_model_catalog(catalog.clone()),
    );
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );
    let register = || {
        Request::builder()
            .method("POST")
            .uri("/api/models/catalog/register")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"relative_path":"nested/detector.engine"}"#))
            .unwrap()
    };

    let first = app.clone().oneshot(register()).await.unwrap();
    assert_eq!(first.status(), axum::http::StatusCode::OK);
    let first: Value =
        serde_json::from_slice(&to_bytes(first.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(first["created"], true);
    assert_eq!(first["project"]["name"], "detector");
    assert_eq!(first["version"]["input_shape"], "engine-probe-required");
    assert_eq!(first["artifact"]["kind"], "engine");
    assert_eq!(first["artifact"]["status"], "pending");
    assert_eq!(
        first["engine_path"],
        engine.canonicalize().unwrap().to_string_lossy().as_ref()
    );

    let second = app.clone().oneshot(register()).await.unwrap();
    let second: Value =
        serde_json::from_slice(&to_bytes(second.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(second["created"], false);
    assert_eq!(second["artifact"]["id"], first["artifact"]["id"]);

    let artifact_id = first["artifact"]["id"].as_i64().unwrap();
    let metadata = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri(format!("/api/models/artifacts/{artifact_id}/metadata"))
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"recommendation":"recommended","tags":["高精度模型","延迟大"]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(metadata.status(), axum::http::StatusCode::OK);
    let metadata: Value =
        serde_json::from_slice(&to_bytes(metadata.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(metadata["recommendation"], "recommended");
    assert_eq!(metadata["tags"], json!(["高精度模型", "延迟大"]));

    let catalog_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/models/catalog")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(catalog_response.status(), axum::http::StatusCode::OK);
    let catalog_body: Value = serde_json::from_slice(
        &to_bytes(catalog_response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let catalog_model = &catalog_body["root"]["children"][0]["children"][0];
    assert_eq!(catalog_model["recommendation"], "recommended");
    assert_eq!(catalog_model["tags"], json!(["高精度模型", "延迟大"]));

    let invalid = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/catalog/register")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"relative_path":"../outside.engine"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid.status(), axum::http::StatusCode::BAD_REQUEST);
    let invalid: Value =
        serde_json::from_slice(&to_bytes(invalid.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(invalid["code"], "MODEL_CATALOG_PATH_INVALID");

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[derive(Debug)]
struct StartRejectingCatalogAdapter {
    catalog: SqliteModelCatalog,
    rejected_artifact_id: i64,
}

impl PerceptionAdapter for StartRejectingCatalogAdapter {
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
        let active = self
            .catalog
            .active_model()
            .map_err(|error| PerceptionError::new(error.to_string()))?
            .ok_or_else(|| PerceptionError::new("model-not-loaded"))?;
        if active.artifact.id == self.rejected_artifact_id {
            return Err(PerceptionError::new(format!(
                "artifact {} failed startup",
                active.artifact.id
            )));
        }
        Ok(Box::new(TestPerceptionSession))
    }
}

#[derive(Debug)]
struct TestPerceptionSession;

impl PerceptionSession for TestPerceptionSession {
    fn shutdown(&mut self) -> Result<(), PerceptionError> {
        Ok(())
    }
}

#[derive(Debug, Default)]
struct BlockingPreflightAdapter {
    entered: AtomicBool,
    release: AtomicBool,
    ingress: Mutex<Option<PipelineIngress>>,
}

impl BlockingPreflightAdapter {
    async fn wait_until_entered(&self) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while !self.entered.load(Ordering::Acquire) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("preflight entered");
    }

    fn trigger_active(&self) -> bool {
        self.ingress
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .is_some_and(PipelineIngress::trigger_active)
    }
}

impl PerceptionAdapter for BlockingPreflightAdapter {
    fn preflight(&self) -> Result<(), PerceptionError> {
        self.entered.store(true, Ordering::Release);
        while !self.release.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }

    fn start(
        &self,
        _epoch: RuntimeEpoch,
        ingress: PipelineIngress,
        _clock: Arc<dyn Clock>,
        _events: std::sync::mpsc::SyncSender<PerceptionEvent>,
    ) -> Result<Box<dyn PerceptionSession>, PerceptionError> {
        *self
            .ingress
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(ingress);
        Ok(Box::new(TestPerceptionSession))
    }
}

fn seed_publishable_artifacts(database: &std::path::Path) {
    let model_directory = database.parent().unwrap().join("models/detector/v1");
    std::fs::create_dir_all(&model_directory).unwrap();
    let artifacts = [
        ("model.engine", b"engine-one".as_slice()),
        ("replacement.engine", b"engine-two".as_slice()),
    ];
    for (name, bytes) in artifacts {
        let checksum = format!("{:x}", Sha256::digest(bytes));
        std::fs::write(model_directory.join(name), bytes).unwrap();
        let manifest = json!({
            "schema_version": 1,
            "model_id": format!("sha256:{checksum}"),
            "display_name": "detector",
            "artifact": {"engine_path": name, "sha256": checksum, "size_bytes": bytes.len()},
            "input": {"name":"images", "shape":[1,3,640,640], "dtype":"float32"},
            "output": {"name":"output0", "shape":[1,5,8400], "dtype":"float32", "class_count":1, "class_names":["target"]},
            "validated": true,
            "model_profile": {
                "status": "VALIDATED",
                "validation": {
                    "status": "validated",
                    "engine_execution_ok": true,
                    "decoder_ok": true,
                    "nms_ok": true,
                    "detection_batch_ok": true,
                    "profile_fingerprint": "sha256:test"
                }
            }
        });
        std::fs::write(
            model_directory.join(format!("{name}.manifest.json")),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }
    let connection = Connection::open(database).unwrap();
    connection
        .execute(
            "INSERT INTO model_projects(name, description) VALUES ('detector', 'primary')",
            [],
        )
        .unwrap();
    connection.execute("INSERT INTO model_versions(project_id, version, source_kind, source_path, classes_json, input_shape) VALUES (1, 'v1', 'onnx', 'model.onnx', '[\"target\"]', '1x3x640x640')", []).unwrap();
    connection.execute("INSERT INTO model_artifacts(version_id, kind, path, checksum, status) VALUES (1, 'engine', 'model.engine', ?1, 'ready')", [format!("sha256:{:x}", Sha256::digest(b"engine-one"))]).unwrap();
    connection.execute("INSERT INTO model_artifacts(version_id, kind, path, checksum, status) VALUES (1, 'engine', 'replacement.engine', ?1, 'ready')", [format!("sha256:{:x}", Sha256::digest(b"engine-two"))]).unwrap();
}

#[tokio::test]
async fn publish_accepts_validated_runtime_manifest_without_legacy_model_profile() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-receipt-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    seed_publishable_artifacts(&database);
    let manifest_path = directory.join("models/detector/v1/model.engine.manifest.json");
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest.as_object_mut().unwrap().remove("model_profile");
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    Connection::open(&database)
        .unwrap()
        .execute(
            "UPDATE model_artifacts SET checksum = 'deferred:test', status = 'pending' WHERE id = 1",
            [],
        )
        .unwrap();
    let dependencies =
        RuntimeDependencies::recording().with_perception(Arc::new(CatalogAwareAdapter {
            catalog: catalog.clone(),
            rejected_artifact_id: 999,
        }));
    let (supervisor, runtime) =
        RuntimeSupervisor::spawn(dependencies.with_model_catalog(catalog.clone()));
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/projects/1/publish")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"artifact_id":1,"parser_preset":"auto"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let active = catalog.active_model().unwrap().unwrap();
    assert_eq!(active.artifact.id, 1);
    assert_eq!(active.artifact.status, "ready");
    assert!(active.artifact.checksum.starts_with("sha256:"));

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn runtime_snapshot_restores_the_active_model_from_the_catalog_on_boot() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-bootstrap-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    seed_publishable_artifacts(&database);
    catalog.publish(1, 1).unwrap();

    let (supervisor, runtime) = RuntimeSupervisor::spawn(
        RuntimeDependencies::recording().with_model_catalog(catalog.clone()),
    );
    let snapshot = runtime.snapshot();
    let active = snapshot.model.active.as_ref().unwrap();
    assert_eq!(active.project.name, "detector");
    assert_eq!(active.version.version, "v1");
    assert_eq!(active.artifact.id, 1);
    assert_eq!(snapshot.model.catalog_error, None);

    let state = get_json(
        build_control_router_with_control_plane(runtime.clone(), None, None, catalog, false, None),
        "/api/runtime/state",
    )
    .await;
    assert_eq!(state["active_model"]["artifact"]["id"], 1);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn publish_while_stopped_preflights_and_commits_without_starting_runtime() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-publish-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    seed_publishable_artifacts(&database);
    let dependencies =
        RuntimeDependencies::recording().with_perception(Arc::new(CatalogAwareAdapter {
            catalog: catalog.clone(),
            rejected_artifact_id: 999,
        }));
    let (supervisor, runtime) =
        RuntimeSupervisor::spawn(dependencies.with_model_catalog(catalog.clone()));
    let config = ConfigService::new(directory.join("novasight.yaml"), AppConfig::default());
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        config,
        None,
        catalog.clone(),
        false,
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/projects/1/publish")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"artifact_id":1,"parser_preset":"yolov8"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["deployment"]["artifact_id"], 1);
    assert_eq!(body["report"]["applied"], true);
    assert_eq!(body["report"]["rolled_back"], false);
    assert_eq!(body["inference"]["selected"], "unconfigured");
    assert_eq!(body["report"]["backend"], "unconfigured");
    assert_eq!(
        runtime.snapshot().pipeline.state,
        novasight_runtime::PipelineState::Stopped
    );
    assert_eq!(catalog.active_model().unwrap().unwrap().artifact.id, 1);
    assert_eq!(
        runtime
            .snapshot()
            .model
            .active
            .as_ref()
            .map(|active| active.artifact.id),
        Some(1)
    );
    let state = get_json(
        build_control_router_with_control_plane(
            runtime.clone(),
            None,
            None,
            catalog.clone(),
            false,
            None,
        ),
        "/api/runtime/state",
    )
    .await;
    assert_eq!(state["active_model"]["project"]["name"], "detector");
    assert_eq!(state["active_model"]["version"]["version"], "v1");
    assert_eq!(state["active_model"]["artifact"]["id"], 1);
    assert_eq!(state["model_catalog_error"], Value::Null);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn publish_without_perception_preflight_fails_closed_before_mutation() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-no-preflight-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    seed_publishable_artifacts(&database);
    let (supervisor, runtime) = RuntimeSupervisor::spawn(
        RuntimeDependencies::recording().with_model_catalog(catalog.clone()),
    );
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/projects/1/publish")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"artifact_id":1,"parser_preset":"auto"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["code"], "MODEL_PREFLIGHT_UNAVAILABLE");
    assert!(catalog.active_model().unwrap().is_none());

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn failed_first_publish_is_rejected_before_deployment_mutation() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-failed-publish-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    seed_publishable_artifacts(&database);
    let dependencies = RuntimeDependencies::new(
        Arc::new(TestClock),
        Arc::new(RecordingPointerDevice::default()),
        PipelineConfig::default(),
    )
    .with_perception(Arc::new(RejectingModelAdapter));
    let (supervisor, runtime) =
        RuntimeSupervisor::spawn(dependencies.with_model_catalog(catalog.clone()));
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/projects/1/publish")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"artifact_id":1,"parser_preset":"auto"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        axum::http::StatusCode::UNPROCESSABLE_ENTITY
    );
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["code"], "MODEL_ACTIVATION_FAILED");
    assert!(
        body["message"]
            .as_str()
            .unwrap()
            .contains("candidate validation failed")
    );
    assert!(catalog.active_model().unwrap().is_none());

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn failed_online_preflight_leaves_previous_model_and_epoch_untouched() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-online-recovery-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    seed_publishable_artifacts(&database);
    catalog.publish(1, 1).unwrap();
    let dependencies = RuntimeDependencies::new(
        Arc::new(TestClock),
        Arc::new(RecordingPointerDevice::default()),
        PipelineConfig::default(),
    )
    .with_perception(Arc::new(CatalogAwareAdapter {
        catalog: catalog.clone(),
        rejected_artifact_id: 2,
    }));
    let (supervisor, runtime) =
        RuntimeSupervisor::spawn(dependencies.with_model_catalog(catalog.clone()));
    runtime.start().await.expect("start previous model");
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/projects/1/publish")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"artifact_id":2,"parser_preset":"auto"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        axum::http::StatusCode::UNPROCESSABLE_ENTITY
    );
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["code"], "MODEL_ACTIVATION_FAILED");
    assert!(
        body["message"]
            .as_str()
            .unwrap()
            .contains("candidate validation failed")
    );
    assert_eq!(catalog.active_model().unwrap().unwrap().artifact.id, 1);
    assert_eq!(
        runtime.snapshot().pipeline.state,
        novasight_runtime::PipelineState::Running
    );
    assert_eq!(runtime.snapshot().pipeline.epoch.unwrap().0, 1);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn failed_online_start_compensates_deployment_and_recovers_previous_epoch() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-online-start-recovery-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    seed_publishable_artifacts(&database);
    catalog.publish(1, 1).unwrap();
    let dependencies = RuntimeDependencies::new(
        Arc::new(TestClock),
        Arc::new(RecordingPointerDevice::default()),
        PipelineConfig::default(),
    )
    .with_perception(Arc::new(StartRejectingCatalogAdapter {
        catalog: catalog.clone(),
        rejected_artifact_id: 2,
    }))
    .with_model_catalog(catalog.clone());
    let (supervisor, runtime) = RuntimeSupervisor::spawn(dependencies);
    runtime.start().await.expect("start previous model");
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/projects/1/publish")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"artifact_id":2,"parser_preset":"auto"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        axum::http::StatusCode::UNPROCESSABLE_ENTITY
    );
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["code"], "MODEL_ACTIVATION_FAILED");
    assert!(
        body["message"]
            .as_str()
            .unwrap()
            .contains("deployment was restored")
    );
    assert_eq!(catalog.active_model().unwrap().unwrap().artifact.id, 1);
    assert_eq!(
        runtime.snapshot().pipeline.state,
        novasight_runtime::PipelineState::Running
    );
    assert_eq!(runtime.snapshot().pipeline.epoch.unwrap().0, 3);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn online_publish_reports_success_only_after_the_new_epoch_is_ready() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-online-publish-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    seed_publishable_artifacts(&database);
    catalog.publish(1, 1).unwrap();
    let dependencies = RuntimeDependencies::new(
        Arc::new(TestClock),
        Arc::new(RecordingPointerDevice::default()),
        PipelineConfig::default(),
    )
    .with_perception(Arc::new(CatalogAwareAdapter {
        catalog: catalog.clone(),
        rejected_artifact_id: 999,
    }));
    let (supervisor, runtime) =
        RuntimeSupervisor::spawn(dependencies.with_model_catalog(catalog.clone()));
    runtime.start().await.expect("start previous model");
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/projects/1/publish")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"artifact_id":2,"parser_preset":"auto"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["deployment"]["artifact_id"], 2);
    assert_eq!(body["report"]["applied"], true);
    assert_eq!(body["inference"]["loaded"], true);
    assert!(
        body["report"]["message"]
            .as_str()
            .unwrap()
            .contains("DetectionBatch")
    );
    assert_eq!(catalog.active_model().unwrap().unwrap().artifact.id, 2);
    assert_eq!(
        runtime.snapshot().pipeline.state,
        novasight_runtime::PipelineState::Running
    );
    assert_eq!(runtime.snapshot().pipeline.epoch.unwrap().0, 2);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn rollback_uses_the_same_preflight_and_compensating_transaction_path() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-rollback-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    seed_publishable_artifacts(&database);
    catalog.publish(1, 1).unwrap();
    catalog.publish(1, 2).unwrap();
    let dependencies = RuntimeDependencies::new(
        Arc::new(TestClock),
        Arc::new(RecordingPointerDevice::default()),
        PipelineConfig::default(),
    )
    .with_perception(Arc::new(CatalogAwareAdapter {
        catalog: catalog.clone(),
        rejected_artifact_id: 999,
    }));
    let (supervisor, runtime) =
        RuntimeSupervisor::spawn(dependencies.with_model_catalog(catalog.clone()));
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/projects/1/rollback")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["report"]["action"], "rollback");
    assert_eq!(body["deployment"]["artifact_id"], 1);
    assert_eq!(body["report"]["applied"], true);
    assert_eq!(catalog.active_model().unwrap().unwrap().artifact.id, 1);
    assert_eq!(
        runtime.snapshot().pipeline.state,
        novasight_runtime::PipelineState::Stopped
    );

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn rollback_without_previous_artifact_preserves_legacy_no_op_success() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-rollback-noop-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    seed_publishable_artifacts(&database);
    let deployment = catalog.publish(1, 1).unwrap();
    let (supervisor, runtime) = RuntimeSupervisor::spawn(
        RuntimeDependencies::recording().with_model_catalog(catalog.clone()),
    );
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/projects/1/rollback")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["deployment"]["artifact_id"], 1);
    assert!(
        body["report"]["message"]
            .as_str()
            .unwrap()
            .contains("没有可回滚")
    );
    assert_eq!(body["report"]["applied"], false);
    assert_eq!(body["report"]["sections"][0]["status"], "unchanged");
    assert_eq!(body["report"]["sections"][1]["status"], "unchanged");
    assert_eq!(body["preparation"]["manifest_action"], "unchanged");
    assert_eq!(catalog.deployment(1).unwrap(), deployment);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_start_cannot_interleave_with_model_activation() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-serialized-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    seed_publishable_artifacts(&database);
    let adapter = Arc::new(BlockingPreflightAdapter::default());
    let dependencies = RuntimeDependencies::new(
        Arc::new(TestClock),
        Arc::new(RecordingPointerDevice::default()),
        PipelineConfig::default(),
    )
    .with_perception(adapter.clone());
    let (supervisor, runtime) =
        RuntimeSupervisor::spawn(dependencies.with_model_catalog(catalog.clone()));
    let app =
        build_control_router_with_control_plane(runtime.clone(), None, None, catalog, false, None);

    let publish_app = app.clone();
    let publish = tokio::spawn(async move {
        publish_app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/models/projects/1/publish")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"artifact_id":1,"parser_preset":"auto"}"#))
                    .unwrap(),
            )
            .await
            .unwrap()
    });
    adapter.wait_until_entered().await;

    let start = tokio::spawn(async move {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/runtime/start")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        !start.is_finished(),
        "runtime start must wait for model activation"
    );

    adapter.release.store(true, Ordering::Release);
    assert_eq!(publish.await.unwrap().status(), axum::http::StatusCode::OK);
    assert_eq!(start.await.unwrap().status(), axum::http::StatusCode::OK);
    assert_eq!(
        runtime.snapshot().pipeline.state,
        novasight_runtime::PipelineState::Running
    );

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overlapping_urgent_stops_keep_activation_cancelled_until_both_are_acknowledged() {
    let directory = std::env::temp_dir().join(format!(
        "novasight-model-emergency-cancel-api-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    seed_publishable_artifacts(&database);
    catalog.publish(1, 1).unwrap();
    let adapter = Arc::new(BlockingPreflightAdapter::default());
    let dependencies = RuntimeDependencies::new(
        Arc::new(TestClock),
        Arc::new(RecordingPointerDevice::default()),
        PipelineConfig::default(),
    )
    .with_perception(adapter.clone())
    .with_model_catalog(catalog.clone());
    let (supervisor, runtime) = RuntimeSupervisor::spawn(dependencies);
    runtime.start().await.unwrap();
    runtime.set_trigger_active(true).await.unwrap();
    assert!(adapter.trigger_active());
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );

    let publish = tokio::spawn(async move {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/models/projects/1/publish")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"artifact_id":2,"parser_preset":"auto"}"#))
                .unwrap(),
        )
        .await
        .unwrap()
    });
    adapter.wait_until_entered().await;

    let emergency_runtime = runtime.clone();
    let emergency = tokio::spawn(async move { emergency_runtime.emergency_stop().await });
    tokio::time::timeout(Duration::from_secs(1), async {
        while adapter.trigger_active() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("urgent stop closes the output gate without waiting for preflight");
    assert_eq!(catalog.active_model().unwrap().unwrap().artifact.id, 1);

    let start_runtime = runtime.clone();
    let start = async move { start_runtime.start().await };
    tokio::pin!(start);
    let start_waker = futures_util::task::noop_waker();
    let mut start_context = std::task::Context::from_waker(&start_waker);
    assert!(
        std::future::Future::poll(start.as_mut(), &mut start_context).is_pending(),
        "start must be queued behind the blocked activation before stop is submitted"
    );
    let stop_runtime = runtime.clone();
    let stop = tokio::spawn(async move { stop_runtime.stop().await });
    tokio::time::timeout(Duration::from_secs(1), async {
        while runtime.pending_urgent_stop_count() < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("both urgent stop requests registered before releasing activation");

    adapter.release.store(true, Ordering::Release);
    let publish_response = publish.await.unwrap();
    assert_eq!(
        publish_response.status(),
        axum::http::StatusCode::UNPROCESSABLE_ENTITY
    );
    let stopped = emergency.await.unwrap().unwrap();
    assert_eq!(
        stopped.pipeline.state,
        novasight_runtime::PipelineState::Stopped
    );
    let start_error = start.await.unwrap_err();
    assert_eq!(
        start_error.kind,
        novasight_runtime::RuntimeErrorKind::InvalidPipelineState
    );
    assert_eq!(
        stop.await.unwrap().unwrap().pipeline.state,
        novasight_runtime::PipelineState::Stopped
    );
    assert_eq!(catalog.active_model().unwrap().unwrap().artifact.id, 1);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn model_routes_return_rows_from_the_existing_database() {
    let directory =
        std::env::temp_dir().join(format!("novasight-model-api-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    let database = directory.join("novasight.db");
    let catalog = SqliteModelCatalog::open(&database).unwrap();
    let model_directory = directory.join("models/detector/v1");
    std::fs::create_dir_all(&model_directory).unwrap();
    std::fs::write(model_directory.join("model.engine"), b"engine").unwrap();
    let connection = Connection::open(&database).unwrap();
    connection
        .execute(
            "INSERT INTO model_projects(name, description) VALUES ('detector', 'primary')",
            [],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO model_projects(name, description) VALUES ('secondary', 'other')",
            [],
        )
        .unwrap();
    connection.execute("INSERT INTO model_versions(project_id, version, source_kind, source_path, classes_json, input_shape) VALUES (1, 'v1', 'onnx', 'model.onnx', '[\"target\"]', '1x3x640x640')", []).unwrap();
    connection.execute("INSERT INTO model_versions(project_id, version, source_kind, source_path, classes_json, input_shape) VALUES (2, 'v2', 'onnx', 'other.onnx', '[\"other\"]', '1x3x320x320')", []).unwrap();
    connection.execute("INSERT INTO model_artifacts(version_id, kind, path, checksum, status) VALUES (1, 'engine', 'model.engine', 'sha256:abc', 'ready')", []).unwrap();
    connection.execute("INSERT INTO model_artifacts(version_id, kind, path, checksum, status) VALUES (1, 'engine', 'missing.engine', 'sha256:missing', 'ready')", []).unwrap();
    connection.execute("INSERT INTO conversion_jobs(version_id, target_kind, command_json, status, log) VALUES (1, 'engine', '[\"trtexec\",\"--onnx=model.onnx\"]', 'succeeded', 'done')", []).unwrap();
    connection.execute("INSERT INTO conversion_jobs(version_id, target_kind, command_json, status, log) VALUES (2, 'engine', '[\"trtexec\",\"--onnx=other.onnx\"]', 'succeeded', 'done')", []).unwrap();
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        None,
        catalog.clone(),
        false,
        None,
    );

    let projects = get_json(app.clone(), "/api/models/projects").await;
    assert_eq!(projects[0]["name"], "detector");
    let versions = get_json(app.clone(), "/api/models/projects/1/versions").await;
    assert_eq!(versions.as_array().unwrap().len(), 1);
    assert_eq!(versions[0]["classes"][0], "target");
    let artifacts = get_json(app.clone(), "/api/models/versions/1/artifacts").await;
    assert_eq!(artifacts[0]["path"], "model.engine");
    assert_eq!(artifacts[0]["size_bytes"], 6);
    assert_eq!(artifacts[1]["size_bytes"], Value::Null);
    let model_catalog = get_json(app.clone(), "/api/models/catalog?force=true").await;
    assert_eq!(model_catalog["model_count"], 1);
    assert_eq!(model_catalog["force"], true);
    assert_eq!(model_catalog["root"]["children"][0]["type"], "directory");
    assert_eq!(
        model_catalog["root"]["children"][0]["children"][0]["children"][0]["artifact_id"],
        1
    );
    let jobs = get_json(app, "/api/models/jobs").await;
    assert_eq!(jobs.as_array().unwrap().len(), 2);
    assert_eq!(jobs[0]["command"][0], "trtexec");

    let filtered_get_jobs = get_json(
        build_control_router_with_control_plane(
            runtime.clone(),
            None,
            None,
            catalog.clone(),
            false,
            None,
        ),
        "/api/models/jobs?version_id=2",
    )
    .await;
    assert_eq!(filtered_get_jobs.as_array().unwrap().len(), 1);
    assert_eq!(filtered_get_jobs[0]["version_id"], 2);

    let filtered_jobs = post_json(
        build_control_router_with_control_plane(
            runtime.clone(),
            None,
            None,
            catalog.clone(),
            false,
            None,
        ),
        "/api/models/jobs/list",
        r#"{"version_id":1}"#,
    )
    .await;
    assert_eq!(filtered_jobs.as_array().unwrap().len(), 1);
    assert_eq!(filtered_jobs[0]["version_id"], 1);

    let unfiltered_jobs = post_json(
        build_control_router_with_control_plane(
            runtime.clone(),
            None,
            None,
            catalog.clone(),
            false,
            None,
        ),
        "/api/models/jobs/list",
        r#"{"version_id":null}"#,
    )
    .await;
    assert_eq!(unfiltered_jobs.as_array().unwrap().len(), 2);

    let response =
        build_control_router_with_control_plane(runtime.clone(), None, None, catalog, false, None)
            .oneshot(
                Request::builder()
                    .uri("/api/models/projects/999/versions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);

    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

async fn post_json(app: axum::Router, path: &str, body: &'static str) -> Value {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(response.status().is_success());
    serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
}

async fn get_json(app: axum::Router, path: &str) -> Value {
    let response = app
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert!(response.status().is_success());
    serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
}
