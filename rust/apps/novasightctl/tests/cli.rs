use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use novasight_api::{
    build_control_router, build_control_router_with_capabilities,
    build_control_router_with_control_plane, build_control_router_with_platform_queries,
    build_control_router_with_services,
};
use novasight_core::{
    CaptureCapabilities, CaptureCapability, CaptureCapabilityProbe, CaptureProbeError,
};
use novasight_runtime::{
    AppConfig, ConfigService, ConfigUpdate, PipelineState, RuntimeDependencies, RuntimeSnapshot,
    RuntimeSupervisor,
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

#[derive(Debug)]
struct CliCaptureProbe;

impl CaptureCapabilityProbe for CliCaptureProbe {
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
        "capture",
    ] {
        assert!(stdout.contains(command), "help omitted {command}");
    }
    assert!(!stdout.contains("diagnose"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capture_commands_use_the_same_platform_query_as_the_web_api() {
    let socket = SocketPath::new();
    let config_path = socket.0.with_extension("yaml");
    std::fs::write(
        &config_path,
        r#"revision: 0
capture:
  device: /dev/video8
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
    let config = ConfigService::new(
        &config_path,
        YamlConfigRepository::load(&config_path).unwrap(),
    );
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind control socket");
    let (supervisor, runtime) = RuntimeSupervisor::spawn_recording();
    let app = build_control_router_with_platform_queries(
        runtime.clone(),
        config,
        None,
        None,
        Arc::new(CliCaptureProbe) as Arc<dyn CaptureCapabilityProbe>,
        false,
        None,
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve control socket")
    });

    let socket_path = socket.0.clone();
    let status =
        tokio::task::spawn_blocking(move || run_cli_args(&socket_path, &["capture", "status"]))
            .await
            .unwrap();
    let status: serde_json::Value = serde_json::from_slice(&status.stdout).unwrap();
    assert_eq!(status["device"], "/dev/video8");

    let socket_path = socket.0.clone();
    let capabilities = tokio::task::spawn_blocking(move || {
        run_cli_args(
            &socket_path,
            &["capture", "capabilities", "--device", "/dev/video3"],
        )
    })
    .await
    .unwrap();
    assert!(
        capabilities.status.success(),
        "{}",
        String::from_utf8_lossy(&capabilities.stderr)
    );
    let capabilities: CaptureCapabilities = serde_json::from_slice(&capabilities.stdout).unwrap();
    assert_eq!(capabilities.device, "/dev/video3");
    assert_eq!(capabilities.capabilities[0].fps_list, vec![120, 60]);

    let socket_path = socket.0.clone();
    let selected = tokio::task::spawn_blocking(move || {
        run_cli_args(
            &socket_path,
            &[
                "capture",
                "select",
                "--device",
                "/dev/video3",
                "--preference",
                "auto_high_fps",
            ],
        )
    })
    .await
    .unwrap();
    assert!(
        selected.status.success(),
        "{}",
        String::from_utf8_lossy(&selected.stderr)
    );
    let selected: serde_json::Value = serde_json::from_slice(&selected.stdout).unwrap();
    assert_eq!(selected["device"], "/dev/video3");
    assert_eq!(selected["profile"]["pixel_format"], "MJPG");
    assert_eq!(selected["profile"]["fps"], 120);

    server.abort();
    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_file(config_path).unwrap();
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
    let rejected = tokio::task::spawn_blocking(move || run_cli_args(&socket_path, &["status"]))
        .await
        .unwrap();
    assert!(!rejected.status.success());
    let rejected = String::from_utf8(rejected.stderr).unwrap();
    assert!(rejected.contains("LICENSE_REQUIRED"));
    assert!(rejected.contains("a valid license is required"));
    assert!(!rejected.contains("daemon_error_response_invalid"));

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
    std::fs::write(
        &config_path,
        "revision: 0\ncontrol:\n  output_enabled: true\nhardware: {}\n",
    )
    .unwrap();
    let initial = YamlConfigRepository::load(&config_path).unwrap();
    let output_enabled = initial.control.output_enabled;
    let config = ConfigService::new(&config_path, initial);
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind control socket");
    let (supervisor, runtime) = RuntimeSupervisor::spawn(
        RuntimeDependencies::recording().with_output_enabled(output_enabled),
    );
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

    runtime.start().await.unwrap();
    runtime.set_trigger_active(true).await.unwrap();
    let socket_path = socket.0.clone();
    let buttons =
        tokio::task::spawn_blocking(move || run_cli_args(&socket_path, &["device", "buttons"]))
            .await
            .unwrap();
    assert!(
        buttons.status.success(),
        "{}",
        String::from_utf8_lossy(&buttons.stderr)
    );
    let buttons: serde_json::Value = serde_json::from_slice(&buttons.stdout).unwrap();
    assert_eq!(buttons["available"], true);
    assert_eq!(buttons["left"], true);
    assert_eq!(buttons["right"], false);
    assert_eq!(buttons["managed_by_runtime"], true);
    runtime.stop().await.unwrap();

    server.abort();
    runtime.shutdown_daemon().await.unwrap();
    supervisor.join().await.unwrap();
    std::fs::remove_file(config_path).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_profile_command_uses_the_daemon_model_contract() {
    let socket = SocketPath::new();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind control socket");
    let app = axum::Router::new()
        .route(
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
        )
        .route(
            "/api/models/artifacts/{artifact_id}/deepstream/recommendation",
            axum::routing::get(
                |axum::extract::Path(artifact_id): axum::extract::Path<i64>| async move {
                    axum::Json(serde_json::json!({
                        "artifact_id": artifact_id,
                        "recommendation": {"input_name": "images", "output_name": "output0"}
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

    let socket_path = socket.0.clone();
    let output = tokio::task::spawn_blocking(move || {
        run_cli_args(&socket_path, &["model", "recommend", "7"])
    })
    .await
    .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let body: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(body["artifact_id"], 7);
    assert_eq!(body["recommendation"]["input_name"], "images");

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_register_command_uses_the_shared_catalog_contract() {
    let socket = SocketPath::new();
    let listener = tokio::net::UnixListener::bind(&socket.0).expect("bind control socket");
    let app = axum::Router::new().route(
        "/api/models/catalog/register",
        axum::routing::post(
            |axum::Json(request): axum::Json<serde_json::Value>| async move {
                assert_eq!(request["relative_path"], "nested/detector.engine");
                axum::Json(serde_json::json!({
                    "project": {"id": 4, "name": "detector", "description": "external"},
                    "version": {
                        "id": 7,
                        "project_id": 4,
                        "version": "external-abc123",
                        "source_kind": "onnx",
                        "source_path": "/srv/models/nested/detector.engine",
                        "classes": ["target"],
                        "input_shape": "engine-probe-required"
                    },
                    "artifact": {
                        "id": 9,
                        "version_id": 7,
                        "kind": "engine",
                        "path": "/srv/models/nested/detector.engine",
                        "checksum": "deferred:abc123",
                        "status": "pending",
                        "size_bytes": 4096
                    },
                    "engine_path": "/srv/models/nested/detector.engine",
                    "created": true
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
            &["model", "register", "nested/detector.engine"],
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
    assert_eq!(body["created"], true);
    assert_eq!(body["project"]["id"], 4);
    assert_eq!(body["artifact"]["id"], 9);

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
