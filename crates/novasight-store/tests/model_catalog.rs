use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use novasight_store::model_catalog::{
    ModelCatalogError, ModelIngressCatalogUpdate, ModelRecommendation, SqliteModelCatalog,
};
use rusqlite::Connection;
use sha2::{Digest, Sha256};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let unique = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "novasight-model-catalog-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn sqlite_data_version(connection: &Connection) -> i64 {
    connection
        .query_row("PRAGMA data_version", [], |row| row.get(0))
        .unwrap()
}

fn create_python_compatible_database(path: &std::path::Path) {
    let connection = Connection::open(path).unwrap();
    connection
        .execute_batch(
            r#"
            PRAGMA foreign_keys = ON;
            CREATE TABLE model_projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL
            );
            CREATE TABLE model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES model_projects(id),
                version TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_path TEXT NOT NULL,
                classes_json TEXT NOT NULL,
                input_shape TEXT NOT NULL,
                UNIQUE(project_id, version)
            );
            CREATE TABLE model_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id INTEGER NOT NULL REFERENCES model_versions(id),
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE conversion_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id INTEGER NOT NULL REFERENCES model_versions(id),
                target_kind TEXT NOT NULL,
                command_json TEXT NOT NULL,
                status TEXT NOT NULL,
                log TEXT NOT NULL
            );
            CREATE TABLE deployments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL UNIQUE REFERENCES model_projects(id),
                artifact_id INTEGER NOT NULL REFERENCES model_artifacts(id),
                previous_artifact_id INTEGER REFERENCES model_artifacts(id),
                updated_seq INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE registry_sequence (name TEXT PRIMARY KEY, value INTEGER NOT NULL);

            INSERT INTO model_projects VALUES (1, 'yolo', 'primary detector');
            INSERT INTO model_versions VALUES
                (2, 1, 'v1', 'onnx', 'source.onnx', '["person","car"]', '1x3x640x640');
            INSERT INTO model_artifacts VALUES
                (3, 2, 'engine', 'model.engine', 'sha256:abc', 'ready');
            INSERT INTO model_artifacts VALUES
                (6, 2, 'engine', 'model-v2.engine', 'sha256:def', 'ready');
            INSERT INTO conversion_jobs VALUES
                (4, 2, 'engine', '["trtexec","--fp16"]', 'succeeded', 'ok');
            INSERT INTO deployments VALUES (5, 1, 3, NULL, 7);
            INSERT INTO registry_sequence VALUES ('deployment', 7);
            "#,
        )
        .unwrap();
}

#[test]
fn catalog_engine_registration_is_reference_only_and_idempotent() {
    let directory = TestDirectory::new();
    let model_root = directory.0.join("models");
    fs::create_dir_all(model_root.join("nested")).unwrap();
    let engine = model_root.join("nested/detector.engine");
    fs::write(&engine, b"opaque-tensorrt-engine").unwrap();
    let catalog =
        SqliteModelCatalog::open_with_model_root(directory.0.join("novasight.db"), &model_root)
            .unwrap();

    let created = catalog
        .register_catalog_engine("nested/detector.engine")
        .unwrap();
    assert!(created.created);
    assert_eq!(created.project.name, "detector");
    assert_eq!(created.version.source_kind, "onnx");
    assert_eq!(created.version.input_shape, "engine-probe-required");
    assert_eq!(created.artifact.kind, "engine");
    assert_eq!(created.artifact.status, "pending");
    assert!(created.artifact.checksum.starts_with("deferred:"));
    assert_eq!(created.artifact.size_bytes, Some(22));
    assert_eq!(created.engine_path, engine.canonicalize().unwrap());

    let reused = catalog
        .register_catalog_engine("nested/detector.engine")
        .unwrap();
    assert!(!reused.created);
    assert_eq!(reused.project.id, created.project.id);
    assert_eq!(reused.version.id, created.version.id);
    assert_eq!(reused.artifact.id, created.artifact.id);
    assert_eq!(catalog.list_projects().unwrap().len(), 1);
    assert_eq!(catalog.list_versions(created.project.id).unwrap().len(), 1);
    assert_eq!(catalog.list_artifacts(created.version.id).unwrap().len(), 1);
    assert_eq!(fs::read(engine).unwrap(), b"opaque-tensorrt-engine");
}

#[test]
fn artifact_metadata_persists_recommendation_and_tags_in_the_catalog_view() {
    let directory = TestDirectory::new();
    let model_root = directory.0.join("models");
    let database_path = directory.0.join("novasight.db");
    fs::create_dir_all(&model_root).unwrap();
    fs::write(model_root.join("detector.engine"), b"engine").unwrap();
    let catalog = SqliteModelCatalog::open_with_model_root(&database_path, &model_root).unwrap();
    let artifact = catalog
        .register_catalog_engine("detector.engine")
        .unwrap()
        .artifact;

    let before = catalog.catalog(false).unwrap();
    assert_eq!(
        before.root.children[0].model().recommendation,
        ModelRecommendation::Unrated
    );
    assert!(before.root.children[0].model().tags.is_empty());

    let metadata = catalog
        .update_artifact_metadata(
            artifact.id,
            ModelRecommendation::Recommended,
            vec![
                " 高精度模型 ".to_owned(),
                "延迟大".to_owned(),
                "高精度模型".to_owned(),
            ],
        )
        .unwrap();
    assert_eq!(metadata.recommendation, ModelRecommendation::Recommended);
    assert_eq!(metadata.tags, ["高精度模型", "延迟大"]);
    assert_eq!(catalog.artifact_metadata(artifact.id).unwrap(), metadata);

    let observer = Connection::open(&database_path).unwrap();
    let version_before_noop = sqlite_data_version(&observer);
    let repeated = catalog
        .update_artifact_metadata(
            artifact.id,
            ModelRecommendation::Recommended,
            vec!["高精度模型".to_owned(), "延迟大".to_owned()],
        )
        .unwrap();
    assert_eq!(repeated, metadata);
    assert_eq!(sqlite_data_version(&observer), version_before_noop);

    let refreshed = catalog.catalog(false).unwrap();
    let model = refreshed.root.children[0].model();
    assert_eq!(model.recommendation, ModelRecommendation::Recommended);
    assert_eq!(model.tags, ["高精度模型", "延迟大"]);
}

#[test]
fn catalog_engine_registration_rejects_escape_and_non_engine_paths() {
    let directory = TestDirectory::new();
    let model_root = directory.0.join("models");
    fs::create_dir_all(&model_root).unwrap();
    fs::write(model_root.join("detector.onnx"), b"onnx").unwrap();
    let catalog =
        SqliteModelCatalog::open_with_model_root(directory.0.join("novasight.db"), model_root)
            .unwrap();

    assert!(matches!(
        catalog.register_catalog_engine("../outside.engine"),
        Err(ModelCatalogError::InvalidCatalogModelPath(_))
    ));
    assert!(matches!(
        catalog.register_catalog_engine("detector.onnx"),
        Err(ModelCatalogError::InvalidCatalogModelPath(_))
    ));
    assert!(matches!(
        catalog.register_catalog_engine("missing.engine"),
        Err(ModelCatalogError::CatalogModelNotFound(_))
    ));
}

#[test]
fn publish_and_rollback_are_atomic_and_preserve_the_previous_artifact() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    create_python_compatible_database(&path);
    let catalog = SqliteModelCatalog::open(&path).unwrap();

    let published = catalog.publish(1, 6).unwrap();
    assert_eq!(published.artifact_id, 6);
    assert_eq!(published.previous_artifact_id, Some(3));
    assert_eq!(published.updated_seq, 8);
    assert_eq!(
        catalog.snapshot().unwrap().active_deployment,
        Some(published)
    );
    assert_eq!(catalog.active_model().unwrap().unwrap().artifact.id, 6);

    let rolled_back = catalog.rollback(1).unwrap();
    assert_eq!(rolled_back.artifact_id, 3);
    assert_eq!(rolled_back.previous_artifact_id, Some(6));
    assert_eq!(rolled_back.updated_seq, 9);
    assert_eq!(
        catalog.snapshot().unwrap().active_deployment,
        Some(rolled_back)
    );
    assert_eq!(catalog.active_model().unwrap().unwrap().artifact.id, 3);
}

#[test]
fn failed_first_activation_can_compensate_by_removing_the_new_deployment() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    create_python_compatible_database(&path);
    let connection = Connection::open(&path).unwrap();
    connection
        .execute("DELETE FROM deployments WHERE project_id = 1", [])
        .unwrap();
    drop(connection);
    let catalog = SqliteModelCatalog::open(&path).unwrap();

    let change = catalog.publish_change(1, 3).unwrap();
    assert!(change.before.is_none());
    assert_eq!(catalog.active_model().unwrap().unwrap().artifact.id, 3);

    let restored = catalog.compensate(change).unwrap();
    assert!(restored.is_none());
    assert!(catalog.active_model().unwrap().is_none());
}

#[test]
fn failed_replacement_restores_the_exact_previous_deployment_and_global_order() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    create_python_compatible_database(&path);
    let catalog = SqliteModelCatalog::open(&path).unwrap();

    let change = catalog.publish_change(1, 6).unwrap();
    assert_eq!(change.before.as_ref().unwrap().artifact_id, 3);
    assert_eq!(change.after.artifact_id, 6);

    let restored = catalog.compensate(change.clone()).unwrap().unwrap();
    assert_eq!(restored.artifact_id, 3);
    assert_eq!(restored.previous_artifact_id, None);
    assert_eq!(restored.updated_seq, change.before.unwrap().updated_seq);
    assert_eq!(catalog.active_model().unwrap().unwrap().artifact.id, 3);
}

#[test]
fn compensation_does_not_make_an_older_project_globally_active() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    create_python_compatible_database(&path);
    let connection = Connection::open(&path).unwrap();
    connection
        .execute(
            "INSERT INTO model_projects(id, name, description) VALUES (2, 'secondary', 'other')",
            [],
        )
        .unwrap();
    connection.execute("INSERT INTO model_versions(id, project_id, version, source_kind, source_path, classes_json, input_shape) VALUES (7, 2, 'v1', 'onnx', 'source.onnx', '[]', '1x3x640x640')", []).unwrap();
    connection.execute("INSERT INTO model_artifacts(id, version_id, kind, path, checksum, status) VALUES (8, 7, 'engine', 'secondary.engine', 'sha256:ghi', 'ready')", []).unwrap();
    drop(connection);
    let catalog = SqliteModelCatalog::open(&path).unwrap();
    catalog.publish(2, 8).unwrap();
    assert_eq!(catalog.active_model().unwrap().unwrap().project.id, 2);

    let change = catalog.publish_change(1, 6).unwrap();
    assert_eq!(catalog.active_model().unwrap().unwrap().project.id, 1);
    catalog.compensate(change).unwrap();

    assert_eq!(catalog.active_model().unwrap().unwrap().project.id, 2);
}

#[test]
fn migrates_the_legacy_python_deployment_table_in_place() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            r#"
            CREATE TABLE model_projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, description TEXT NOT NULL);
            CREATE TABLE model_versions (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES model_projects(id), version TEXT NOT NULL, source_kind TEXT NOT NULL, source_path TEXT NOT NULL, classes_json TEXT NOT NULL, input_shape TEXT NOT NULL, UNIQUE(project_id, version));
            CREATE TABLE model_artifacts (id INTEGER PRIMARY KEY, version_id INTEGER NOT NULL REFERENCES model_versions(id), kind TEXT NOT NULL, path TEXT NOT NULL, checksum TEXT NOT NULL, status TEXT NOT NULL);
            CREATE TABLE deployments (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL UNIQUE REFERENCES model_projects(id), artifact_id INTEGER NOT NULL REFERENCES model_artifacts(id), previous_artifact_id INTEGER REFERENCES model_artifacts(id));
            INSERT INTO model_projects VALUES (1, 'legacy', 'legacy db');
            INSERT INTO model_versions VALUES (2, 1, 'v1', 'onnx', 'source.onnx', '[]', '1x3x640x640');
            INSERT INTO model_artifacts VALUES (3, 2, 'engine', 'legacy.engine', 'abc', 'ready');
            INSERT INTO deployments VALUES (5, 1, 3, NULL);
            "#,
        )
        .unwrap();
    drop(connection);

    let snapshot = SqliteModelCatalog::open(&path).unwrap().snapshot().unwrap();

    assert_eq!(snapshot.active_deployment.unwrap().updated_seq, 5);
    let connection = Connection::open(path).unwrap();
    let migration: i64 = connection
        .query_row(
            "SELECT version FROM schema_migrations WHERE name = 'model_catalog'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(migration, 2);
}

#[test]
fn reads_the_existing_python_catalog_without_copying_or_rewriting_it() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    create_python_compatible_database(&path);

    let snapshot = SqliteModelCatalog::open(&path).unwrap().snapshot().unwrap();

    assert_eq!(snapshot.projects.len(), 1);
    assert_eq!(snapshot.projects[0].name, "yolo");
    assert_eq!(snapshot.versions[0].classes, ["person", "car"]);
    assert_eq!(snapshot.artifacts[0].status, "ready");
    assert_eq!(snapshot.jobs[0].command, ["trtexec", "--fp16"]);
    assert_eq!(snapshot.deployments[0].artifact_id, 3);
    assert_eq!(snapshot.active_deployment.unwrap().updated_seq, 7);

    let active = SqliteModelCatalog::open(&path)
        .unwrap()
        .active_model()
        .unwrap()
        .unwrap();
    assert_eq!(active.project.name, "yolo");
    assert_eq!(active.version.classes, ["person", "car"]);
    assert_eq!(active.artifact.id, 3);
    assert_eq!(
        active.artifact_path,
        directory.0.join("models/yolo/v1/model.engine")
    );
}

#[test]
fn filtered_reads_preserve_python_not_found_semantics() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    create_python_compatible_database(&path);
    let catalog = SqliteModelCatalog::open(&path).unwrap();

    assert_eq!(catalog.list_versions(1).unwrap().len(), 1);
    assert!(matches!(
        catalog.list_versions(999),
        Err(ModelCatalogError::ProjectNotFound(999))
    ));
    assert!(matches!(
        catalog.list_artifacts(999),
        Err(ModelCatalogError::VersionNotFound(999))
    ));
    assert!(matches!(
        catalog.list_jobs(Some(999)),
        Err(ModelCatalogError::VersionNotFound(999))
    ));
}

#[test]
fn model_ingress_commit_verifies_engine_identity_and_updates_registry_atomically() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    create_python_compatible_database(&path);
    let engine_dir = directory.0.join("models/yolo/v1");
    fs::create_dir_all(&engine_dir).unwrap();
    let engine = engine_dir.join("model-v2.engine");
    fs::write(&engine, b"engine-v2").unwrap();
    let checksum = format!("sha256:{:x}", Sha256::digest(b"engine-v2"));
    Connection::open(&path)
        .unwrap()
        .execute("DELETE FROM deployments", [])
        .unwrap();
    let catalog = SqliteModelCatalog::open(&path).unwrap();

    let updated = catalog
        .commit_model_ingress(ModelIngressCatalogUpdate {
            artifact_id: 6,
            status: "pending".to_owned(),
            checksum: checksum.clone(),
            classes: Some(vec!["target".to_owned()]),
            input_shape: Some("1x3x320x320".to_owned()),
        })
        .unwrap();

    assert_eq!(updated.artifact.checksum, checksum);
    assert_eq!(updated.artifact.status, "pending");
    assert_eq!(updated.version.classes, ["target"]);
    assert_eq!(updated.version.input_shape, "1x3x320x320");

    let error = catalog
        .commit_model_ingress(ModelIngressCatalogUpdate {
            artifact_id: 6,
            status: "ready".to_owned(),
            checksum: "sha256:wrong".to_owned(),
            classes: None,
            input_shape: None,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        ModelCatalogError::IngressIdentityMismatch { artifact_id: 6, .. }
    ));
    assert_eq!(
        catalog.runtime_artifact_by_id(6).unwrap().artifact.status,
        "pending"
    );
}

#[test]
fn model_ingress_refuses_to_mutate_the_active_artifact() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    create_python_compatible_database(&path);
    let engine_dir = directory.0.join("models/yolo/v1");
    fs::create_dir_all(&engine_dir).unwrap();
    fs::write(engine_dir.join("model.engine"), b"engine").unwrap();
    let checksum = format!("sha256:{:x}", Sha256::digest(b"engine"));
    let catalog = SqliteModelCatalog::open(&path).unwrap();

    assert!(catalog.artifact_is_active(3).unwrap());
    assert!(matches!(
        catalog.commit_model_ingress(ModelIngressCatalogUpdate {
            artifact_id: 3,
            status: "pending".to_owned(),
            checksum,
            classes: None,
            input_shape: None,
        }),
        Err(ModelCatalogError::VersionCurrentlyActive {
            version_id: 2,
            active_artifact_id: 3
        })
    ));

    fs::write(engine_dir.join("model-v2.engine"), b"engine-v2").unwrap();
    let sibling_checksum = format!("sha256:{:x}", Sha256::digest(b"engine-v2"));
    assert!(matches!(
        catalog.commit_model_ingress(ModelIngressCatalogUpdate {
            artifact_id: 6,
            status: "pending".to_owned(),
            checksum: sibling_checksum.clone(),
            classes: Some(vec!["changed".to_owned()]),
            input_shape: Some("1x3x320x320".to_owned()),
        }),
        Err(ModelCatalogError::VersionCurrentlyActive {
            version_id: 2,
            active_artifact_id: 3
        })
    ));

    let safe_sibling_update = catalog
        .commit_model_ingress(ModelIngressCatalogUpdate {
            artifact_id: 6,
            status: "pending".to_owned(),
            checksum: sibling_checksum,
            classes: Some(vec!["person".to_owned(), "car".to_owned()]),
            input_shape: Some("1x3x640x640".to_owned()),
        })
        .expect("a non-deployed sibling with unchanged version metadata is mutable");
    assert_eq!(safe_sibling_update.artifact.status, "pending");
}

#[test]
fn model_ingress_can_refresh_an_inactive_historical_deployment() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    create_python_compatible_database(&path);
    let primary_engine_dir = directory.0.join("models/yolo/v1");
    fs::create_dir_all(&primary_engine_dir).unwrap();
    fs::write(primary_engine_dir.join("model.engine"), b"engine").unwrap();
    let checksum = format!("sha256:{:x}", Sha256::digest(b"engine"));

    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            r#"
            INSERT INTO model_projects VALUES (7, 'active', 'currently active detector');
            INSERT INTO model_versions VALUES
                (8, 7, 'v1', 'onnx', 'active.onnx', '["target"]', '1x3x320x320');
            INSERT INTO model_artifacts VALUES
                (9, 8, 'engine', 'active.engine', 'sha256:active', 'ready');
            INSERT INTO deployments VALUES (10, 7, 9, NULL, 8);
            UPDATE registry_sequence SET value = 8 WHERE name = 'deployment';
            "#,
        )
        .unwrap();
    drop(connection);

    let catalog = SqliteModelCatalog::open(&path).unwrap();
    assert!(
        catalog
            .snapshot()
            .unwrap()
            .deployments
            .iter()
            .any(|deployment| deployment.artifact_id == 3)
    );
    assert!(!catalog.artifact_is_active(3).unwrap());
    assert_eq!(catalog.active_model().unwrap().unwrap().artifact.id, 9);

    let refreshed = catalog
        .commit_model_ingress(ModelIngressCatalogUpdate {
            artifact_id: 3,
            status: "pending".to_owned(),
            checksum,
            classes: Some(vec!["target".to_owned()]),
            input_shape: Some("1x3x256x256".to_owned()),
        })
        .expect("an inactive historical deployment must remain refreshable");

    assert_eq!(refreshed.artifact.status, "pending");
    assert_eq!(refreshed.version.classes, ["target"]);
    assert_eq!(refreshed.version.input_shape, "1x3x256x256");
}

#[test]
fn failed_migration_rolls_back_every_schema_change() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            r#"
            CREATE TABLE model_projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, description TEXT NOT NULL);
            CREATE TABLE model_versions (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES model_projects(id), version TEXT NOT NULL, source_kind TEXT NOT NULL, source_path TEXT NOT NULL, classes_json TEXT NOT NULL, input_shape TEXT NOT NULL, UNIQUE(project_id, version));
            CREATE TABLE model_artifacts (id INTEGER PRIMARY KEY, version_id INTEGER NOT NULL REFERENCES model_versions(id), kind TEXT NOT NULL, path TEXT NOT NULL, checksum TEXT NOT NULL, status TEXT NOT NULL);
            CREATE TABLE deployments (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL UNIQUE REFERENCES model_projects(id), artifact_id INTEGER NOT NULL REFERENCES model_artifacts(id), previous_artifact_id INTEGER REFERENCES model_artifacts(id));
            CREATE TABLE registry_sequence (name TEXT PRIMARY KEY);
            "#,
        )
        .unwrap();
    drop(connection);

    assert!(SqliteModelCatalog::open(&path).is_err());

    let connection = Connection::open(path).unwrap();
    let columns = connection
        .prepare("PRAGMA table_info(deployments)")
        .unwrap()
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    assert!(!columns.iter().any(|column| column == "updated_seq"));
    let migration_table_exists: bool = connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(!migration_table_exists);
}

#[test]
fn snapshot_keeps_deployments_and_active_deployment_in_one_read_transaction() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    create_python_compatible_database(&path);
    let catalog = Arc::new(SqliteModelCatalog::open(&path).unwrap());
    let writer_catalog = Arc::clone(&catalog);
    let writer = std::thread::spawn(move || {
        for _ in 0..50 {
            writer_catalog.publish(1, 6).unwrap();
            writer_catalog.rollback(1).unwrap();
        }
    });

    for _ in 0..100 {
        let snapshot = catalog.snapshot().unwrap();
        let active = snapshot.active_deployment.unwrap();
        assert_eq!(
            snapshot
                .deployments
                .iter()
                .find(|deployment| deployment.id == active.id),
            Some(&active)
        );
    }
    writer.join().unwrap();
}

#[test]
fn catalog_snapshot_is_stable_and_force_refresh_stays_lightweight() {
    let directory = TestDirectory::new();
    let path = directory.0.join("novasight.db");
    let model_root = directory.0.join("models");
    fs::create_dir_all(&model_root).unwrap();
    let engine = model_root.join("model.engine");
    fs::write(&engine, b"engine").unwrap();
    fs::write(
        model_root.join("model.engine.manifest.json"),
        r#"{
            "schema_version": 1,
            "model_id": "detector",
            "display_name": "Detector",
            "artifact": {
                "engine_path": "model.engine",
                "sha256": "ed9f6f25068608efd412958da4dfc19328ca3511251fa6d5f9c42baf230e32f8",
                "size_bytes": 6
            },
            "input": {"name":"images","shape":[1,3,640,640],"dtype":"float32"},
            "output": {"name":"output","shape":[1,84,8400],"dtype":"float32"}
        }"#,
    )
    .unwrap();
    let catalog = SqliteModelCatalog::open_with_model_root(&path, &model_root).unwrap();

    let lightweight = catalog.catalog(false).unwrap();
    let cached = catalog.catalog(false).unwrap();
    let forced = catalog.catalog(true).unwrap();
    assert_eq!(
        lightweight.root.children[0].model().scan_status,
        "need_confirm"
    );
    assert_eq!(cached.cache_hits, 1);
    assert_eq!(forced.root.children[0].model().scan_status, "need_confirm");

    fs::write(&engine, b"corrupt").unwrap();
    fs::write(
        model_root.join("model.engine.manifest.json"),
        b"not-json-and-must-not-be-read-by-catalog-refresh",
    )
    .unwrap();

    let still_cached = catalog.catalog(false).unwrap();
    assert_eq!(still_cached.root.children[0].model().size_bytes, 6);
    let refreshed = catalog.catalog(true).unwrap();
    assert_eq!(refreshed.root.children[0].model().size_bytes, 7);
    assert_eq!(
        refreshed.root.children[0].model().scan_status,
        "need_confirm"
    );
}

trait CatalogNodeTestExt {
    fn model(&self) -> &novasight_store::model_catalog::ModelCatalogModel;
}

impl CatalogNodeTestExt for novasight_store::model_catalog::ModelCatalogNode {
    fn model(&self) -> &novasight_store::model_catalog::ModelCatalogModel {
        match self {
            Self::Model(model) => model,
            Self::Directory(_) => panic!("expected model node"),
        }
    }
}
