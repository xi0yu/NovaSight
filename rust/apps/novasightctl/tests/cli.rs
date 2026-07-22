use std::path::{Path, PathBuf};
use std::process::Command;

use novasight_api::{
    build_control_router, build_control_router_with_capabilities,
    build_control_router_with_control_plane, build_control_router_with_services,
};
use novasight_runtime::{
    AppConfig, ConfigService, ConfigUpdate, PipelineState, RuntimeSnapshot, RuntimeSupervisor,
};
use novasight_store::config::YamlConfigRepository;
use novasight_store::license::{FileLicenseRepository, LicensePolicy, LicenseStatus};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_novasightctl")
}

struct SocketPath(PathBuf);

impl SocketPath {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "novasightctl-{}-{}.sock",
            std::process::id(),
            unique_id()
        )))
    }
}

impl Drop for SocketPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn unique_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[test]
fn help_documents_the_real_command_surface() {
    let output = Command::new(binary())
        .arg("--help")
        .output()
        .expect("run help");
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 help");

    assert!(output.status.success());
    for command in [
        "status",
        "start",
        "stop",
        "restart",
        "emergency-stop",
        "config",
        "license",
        "model",
        "device",
    ] {
        assert!(stdout.contains(command), "help omitted {command}");
    }
    assert!(!stdout.contains("diagnose"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn license_commands_bootstrap_through_a_key_file_and_the_real_repository() {
    let socket = SocketPath::new();
    let license_path = socket.0.with_extension("license.json");
    let key_path = socket.0.with_extension("license.key");
    std::fs::write(&key_path, "NOVASIGHT-TEST-MAX-ACCESS-2026\n").unwrap();
    let repository = FileLicenseRepository::new(&license_path, LicensePolicy::new(true, None));
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind control socket");
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_control_plane(
        runtime.clone(),
        None,
        repository,
        None,
        false,
        None,
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve control socket")
    });

    let socket_path = socket.0.clone();
    let key_path_for_cli = key_path.clone();
    let activated = tokio::task::spawn_blocking(move || {
        run_cli_args(
            &socket_path,
            &[
                "license",
                "activate",
                "--key-file",
                key_path_for_cli.to_str().unwrap(),
            ],
        )
    })
    .await
    .unwrap();
    assert!(
        activated.status.success(),
        "{}",
        String::from_utf8_lossy(&activated.stderr)
    );
    let activated: LicenseStatus = serde_json::from_slice(&activated.stdout).unwrap();
    assert!(activated.configured);
    assert!(activated.valid);
    assert!(
        !std::fs::read_to_string(&license_path)
            .unwrap()
            .contains("NOVASIGHT-TEST-MAX-ACCESS-2026")
    );

    let socket_path = socket.0.clone();
    let status =
        tokio::task::spawn_blocking(move || run_cli_args(&socket_path, &["license", "status"]))
            .await
            .unwrap();
    let status: LicenseStatus = serde_json::from_slice(&status.stdout).unwrap();
    assert!(status.valid);

    let socket_path = socket.0.clone();
    let cleared =
        tokio::task::spawn_blocking(move || run_cli_args(&socket_path, &["license", "clear"]))
            .await
            .unwrap();
    let cleared: LicenseStatus = serde_json::from_slice(&cleared.stdout).unwrap();
    assert!(!cleared.configured);

    server.abort();
    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_file(key_path).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn device_commands_use_the_supervisor_owned_diagnostic_path() {
    let socket = SocketPath::new();
    let config_path = socket.0.with_extension("yaml");
    std::fs::write(&config_path, "revision: 0\nhardware: {}\n").unwrap();
    let config = ConfigService::new(
        &config_path,
        YamlConfigRepository::load(&config_path).unwrap(),
    );
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind control socket");
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_capabilities(runtime.clone(), config, true, None);
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve control socket")
    });

    let socket_path = socket.0.clone();
    let moved = tokio::task::spawn_blocking(move || {
        run_cli_args(&socket_path, &["device", "move", "5", "-3"])
    })
    .await
    .unwrap();
    assert!(
        moved.status.success(),
        "{}",
        String::from_utf8_lossy(&moved.stderr)
    );
    let moved: serde_json::Value = serde_json::from_slice(&moved.stdout).unwrap();
    assert_eq!(moved["sent"], true);
    assert_eq!(moved["receipt"]["delta_x_counts"], 5);
    assert_eq!(moved["receipt"]["delta_y_counts"], -3);

    let socket_path = socket.0.clone();
    let status =
        tokio::task::spawn_blocking(move || run_cli_args(&socket_path, &["device", "status"]))
            .await
            .unwrap();
    let status: serde_json::Value = serde_json::from_slice(&status.stdout).unwrap();
    assert_eq!(status["executors"]["kmnet"]["move_count"], 1);
    assert_eq!(status["executors"]["kmnet"]["last_dx"], 5);

    server.abort();
    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_file(config_path).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_profile_command_uses_the_daemon_model_contract() {
    let socket = SocketPath::new();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind control socket");
    let app = axum::Router::new().route(
        "/api/models/artifacts/{artifact_id}/profile",
        axum::routing::get(
            |axum::extract::Path(artifact_id): axum::extract::Path<i64>| async move {
                axum::Json(serde_json::json!({
                    "artifact_id": artifact_id,
                    "profile_path": "models/detector/v1/model.engine.manifest.json",
                    "profile": {"status": "VALIDATED"}
                }))
            },
        ),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve control socket")
    });

    let socket_path = socket.0.clone();
    let output =
        tokio::task::spawn_blocking(move || run_cli_args(&socket_path, &["model", "profile", "7"]))
            .await
            .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let body: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(body["artifact_id"], 7);
    assert_eq!(body["profile"]["status"], "VALIDATED");

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_publish_command_uses_the_shared_activation_contract() {
    let socket = SocketPath::new();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind control socket");
    let app = axum::Router::new().route(
        "/api/models/projects/{project_id}/publish",
        axum::routing::post(
            |axum::extract::Path(project_id): axum::extract::Path<i64>,
             axum::Json(request): axum::Json<serde_json::Value>| async move {
                assert_eq!(project_id, 4);
                assert_eq!(request["artifact_id"], 9);
                assert_eq!(request["parser_preset"], "yolov8");
                axum::Json(serde_json::json!({
                    "deployment": {
                        "id": 12,
                        "project_id": project_id,
                        "artifact_id": 9,
                        "previous_artifact_id": 8,
                        "updated_seq": 2
                    },
                    "inference": {"loaded": true},
                    "parser_contract": null,
                    "preparation": {
                        "manifest_action": "reused",
                        "reason": "validated",
                        "input_shape": "1x3x640x640",
                        "classes": ["target"]
                    },
                    "report": {
                        "action": "publish",
                        "applied": true,
                        "rolled_back": false,
                        "message": "activated",
                        "runtime_error": "",
                        "artifact_id": 9,
                        "previous_artifact_id": 8,
                        "artifact_path": "models/detector/v2/model.engine",
                        "backend": "deepstream_nvinfer",
                        "input_shape": "1x3x640x640",
                        "classes": 1,
                        "sections": []
                    }
                }))
            },
        ),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve control socket")
    });

    let socket_path = socket.0.clone();
    let output = tokio::task::spawn_blocking(move || {
        run_cli_args(
            &socket_path,
            &["model", "publish", "4", "9", "--parser-preset", "yolov8"],
        )
    })
    .await
    .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let body: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(body["deployment"]["artifact_id"], 9);
    assert_eq!(body["report"]["applied"], true);

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn config_commands_share_the_daemon_revisioned_config_service() {
    let socket = SocketPath::new();
    let config_path = socket.0.with_extension("yaml");
    std::fs::write(&config_path, "revision: 3\nserver:\n  port: 5174\n").unwrap();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind control socket");
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let config = ConfigService::new(
        &config_path,
        YamlConfigRepository::load(&config_path).unwrap(),
    );
    let app = build_control_router_with_services(runtime.clone(), Some(config), None);
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve control socket")
    });

    let socket_path = socket.0.clone();
    let shown =
        tokio::task::spawn_blocking(move || run_cli_args(&socket_path, &["config", "show"]))
            .await
            .unwrap();
    assert!(shown.status.success());
    let shown: AppConfig = serde_json::from_slice(&shown.stdout).unwrap();
    assert_eq!(shown.revision, 3);

    let socket_path = socket.0.clone();
    let updated = tokio::task::spawn_blocking(move || {
        run_cli_args(
            &socket_path,
            &[
                "config",
                "set",
                "server",
                "port",
                "6002",
                "--expected-revision",
                "3",
            ],
        )
    })
    .await
    .unwrap();
    assert!(updated.status.success());
    let updated: ConfigUpdate = serde_json::from_slice(&updated.stdout).unwrap();
    assert_eq!(updated.config.revision, 4);
    assert_eq!(updated.config.server.port, 6002);
    assert!(updated.restart_required);
    assert!(!updated.applied);

    server.abort();
    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_file(config_path).unwrap();
}

#[test]
fn missing_daemon_exits_nonzero_with_a_stable_error_code() {
    let socket = SocketPath::new();
    let output = run_cli(&socket.0, "status");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");

    assert!(!output.status.success());
    assert!(stderr.contains("daemon_connect_failed"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commands_print_the_snapshot_returned_by_the_daemon() {
    let socket = SocketPath::new();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind control socket");
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let server_runtime = runtime.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, build_control_router(server_runtime))
            .await
            .expect("serve control socket")
    });

    let socket_path = socket.0.clone();
    let started = tokio::task::spawn_blocking(move || run_cli(&socket_path, "start"))
        .await
        .expect("join CLI process");
    assert!(started.status.success());
    let snapshot: RuntimeSnapshot = serde_json::from_slice(&started.stdout).expect("snapshot JSON");
    assert_eq!(snapshot.pipeline.state, PipelineState::Running);
    assert_eq!(runtime.snapshot().pipeline.state, PipelineState::Running);

    let socket_path = socket.0.clone();
    let stopped = tokio::task::spawn_blocking(move || run_cli(&socket_path, "emergency-stop"))
        .await
        .expect("join CLI process");
    assert!(stopped.status.success());
    let snapshot: RuntimeSnapshot = serde_json::from_slice(&stopped.stdout).expect("snapshot JSON");
    assert_eq!(snapshot.pipeline.state, PipelineState::Stopped);

    server.abort();
    runtime
        .shutdown_daemon()
        .await
        .expect("shutdown supervisor");
    supervisor.join().await.expect("join supervisor");
}

fn run_cli(socket: &Path, command: &str) -> std::process::Output {
    run_cli_args(socket, &[command])
}

fn run_cli_args(socket: &Path, arguments: &[&str]) -> std::process::Output {
    Command::new(binary())
        .args(["--socket", socket.to_str().expect("UTF-8 socket")])
        .args(arguments)
        .output()
        .expect("run novasightctl")
}
