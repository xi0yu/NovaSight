use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use novasight_store::model_catalog::{ModelCatalogError, SqliteModelCatalog};
use rusqlite::Connection;

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
    assert_eq!(migration, 1);
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
fn catalog_cache_and_force_scan_have_real_distinct_semantics() {
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
    assert_eq!(forced.root.children[0].model().scan_status, "ready");

    let manifest_path = model_root.join("model.engine.manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let mut wrong_filename = manifest.clone();
    wrong_filename["artifact"]["engine_path"] =
        serde_json::Value::String("other.engine".to_owned());
    fs::write(&manifest_path, serde_json::to_vec(&wrong_filename).unwrap()).unwrap();
    assert_eq!(
        catalog.catalog(true).unwrap().root.children[0]
            .model()
            .scan_status,
        "invalid"
    );

    let mut malformed_runtime = manifest.clone();
    malformed_runtime["runtime"] = serde_json::json!({"unexpected": true});
    fs::write(
        &manifest_path,
        serde_json::to_vec(&malformed_runtime).unwrap(),
    )
    .unwrap();
    assert_eq!(
        catalog.catalog(true).unwrap().root.children[0]
            .model()
            .scan_status,
        "invalid"
    );

    let mut wrong_fingerprint = manifest;
    wrong_fingerprint["model_fingerprint"] = serde_json::Value::String("wrong".to_owned());
    fs::write(
        &manifest_path,
        serde_json::to_vec(&wrong_fingerprint).unwrap(),
    )
    .unwrap();
    let invalid_fingerprint = catalog.catalog(true).unwrap();
    assert_eq!(
        invalid_fingerprint.root.children[0].model().scan_status,
        "invalid"
    );

    fs::write(&engine, b"corrupt").unwrap();
    let invalid = catalog.catalog(true).unwrap();
    assert_eq!(invalid.root.children[0].model().scan_status, "invalid");
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
