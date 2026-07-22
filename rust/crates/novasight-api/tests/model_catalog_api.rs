use axum::body::{Body, to_bytes};
use axum::http::Request;
use novasight_api::build_control_router_with_control_plane;
use novasight_runtime::RuntimeSupervisor;
use novasight_store::model_catalog::SqliteModelCatalog;
use rusqlite::Connection;
use serde_json::Value;
use tower::ServiceExt;

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
