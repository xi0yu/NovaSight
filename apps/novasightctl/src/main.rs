//! `novasightctl` — the thin local client for the daemon authority.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use novasight_client::{
    CatalogEngineRegistration, ClientError, ControlClient, DeviceButtons, DiagnosticMoveResponse,
    ExecutorStatus, LicenseStatus, ModelArtifact, ModelProject, ModelSwitchResponse, ModelVersion,
};
use novasight_core::CaptureSelectionPreference;
use novasight_runtime::{
    AppConfig, ConfigUpdate, CrosshairSnapshot, ModelIngressResult, ModelProbeInputMode,
    ModelProfileConfigureRequest, PreviewSnapshot, RuntimeSnapshot,
};
use serde::Serialize;
use thiserror::Error;

const PORTABLE_CONTROL_SOCKET: &str = "run/novasightd.sock";
const CONTROL_SOCKET_ENV: &str = "NOVASIGHT_CONTROL_SOCKET";

#[derive(Parser, Debug)]
#[command(name = "novasightctl", about = "NovaSight daemon command-line client")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print the current immutable runtime snapshot.
    Status,
    /// Start the runtime pipeline.
    Start,
    /// Stop the runtime pipeline.
    Stop,
    /// Restart the runtime pipeline with a new epoch.
    Restart,
    /// Immediately close output and stop the runtime pipeline.
    EmergencyStop,
    /// Read or update the daemon's persisted YAML configuration.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Inspect or update the daemon license through the local control socket.
    License {
        #[command(subcommand)]
        command: LicenseCommand,
    },
    /// Inspect, configure, and diagnostically execute TensorRT model artifacts.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Inspect the runtime-owned pointer device or send one bounded diagnostic move.
    Device {
        #[command(subcommand)]
        command: DeviceCommand,
    },
    /// Inspect configured capture state or enumerate real V4L2 profiles.
    Capture {
        #[command(subcommand)]
        command: CaptureCommand,
    },
    /// Inspect or gate the hardware JPEG preview branch.
    Preview {
        #[command(subcommand)]
        command: PreviewCommand,
    },
    /// Inspect or mutate the daemon-owned visual crosshair template.
    Crosshair {
        #[command(subcommand)]
        command: CrosshairCommand,
    },
}

#[derive(Subcommand, Debug)]
enum CrosshairCommand {
    /// Print observation, template, and control-reference readiness.
    Status,
    /// Learn a template from the configured number of fresh center samples.
    Learn,
    /// Remove the persisted template and return to geometric-center fallback.
    Clear,
}

#[derive(Subcommand, Debug)]
enum PreviewCommand {
    /// Print preview availability, subscribers, and encoder state.
    Status,
    /// Permit preview encoding when at least one stream viewer is connected.
    On,
    /// Close the preview valve and discard the latest JPEG.
    Off,
}

#[derive(Subcommand, Debug)]
enum LicenseCommand {
    /// Print verified license state.
    Status,
    /// Activate a license read from a file so the key is not exposed in argv.
    Activate {
        #[arg(long)]
        key_file: PathBuf,
    },
    /// Remove the currently persisted license.
    Clear,
}

#[derive(Subcommand, Debug)]
enum ModelCommand {
    /// Register an existing .engine below a configured daemon model root.
    Register { relative_path: String },
    /// List model projects in the daemon catalog.
    Projects,
    /// List versions belonging to one project.
    Versions { project_id: i64 },
    /// List artifacts belonging to one model version.
    Artifacts { version_id: i64 },
    /// Deserialize an Engine and write its initial immutable profile.
    Inspect { artifact_id: i64 },
    /// Read the current profile from the unified Engine manifest.
    Profile { artifact_id: i64 },
    /// Derive the DeepStream contract from the bound Engine profile receipt.
    Recommend { artifact_id: i64 },
    /// Apply explicit preprocessing and decoder semantics from a JSON file.
    Configure {
        artifact_id: i64,
        #[arg(long)]
        request: PathBuf,
    },
    /// Execute an isolated candidate TensorRT context without publishing it.
    Probe {
        artifact_id: i64,
        #[arg(long, value_enum, default_value_t = CliProbeInputMode::Fixed)]
        input_mode: CliProbeInputMode,
    },
    /// Atomically publish a validated artifact and restart the active epoch when needed.
    Publish {
        project_id: i64,
        artifact_id: i64,
        #[arg(long, default_value = "auto")]
        parser_preset: String,
    },
    /// Roll back one project through the same compensated activation transaction.
    Rollback { project_id: i64 },
}

#[derive(Subcommand, Debug)]
enum DeviceCommand {
    /// Print the supervisor-owned executor state.
    Status,
    /// Print the latest daemon-owned left/right hardware button state.
    Buttons,
    /// Send exactly one immediate raw move through the daemon-owned adapter.
    Move {
        #[arg(allow_hyphen_values = true)]
        dx: i32,
        #[arg(allow_hyphen_values = true)]
        dy: i32,
    },
}

#[derive(Subcommand, Debug)]
enum CaptureCommand {
    /// Print the supervisor-owned capture state.
    Status,
    /// Enumerate discrete V4L2 formats, sizes, and frame rates.
    Capabilities {
        #[arg(long, default_value = "/dev/video0")]
        device: String,
    },
    /// Select and persist one kernel-reported profile for the next runtime start.
    Select {
        #[arg(long, default_value = "/dev/video0")]
        device: String,
        #[arg(long, value_enum, default_value_t = CliCapturePreference::AutoHighFps)]
        preference: CliCapturePreference,
        #[arg(long, requires_all = ["width", "height", "fps"])]
        pixel_format: Option<String>,
        #[arg(long, requires_all = ["pixel_format", "height", "fps"])]
        width: Option<u32>,
        #[arg(long, requires_all = ["pixel_format", "width", "fps"])]
        height: Option<u32>,
        #[arg(long, requires_all = ["pixel_format", "width", "height"])]
        fps: Option<u32>,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "snake_case")]
enum CliCapturePreference {
    #[default]
    AutoHighFps,
    AutoLowLatency,
    AutoBalanced,
    Manual,
}

impl From<CliCapturePreference> for CaptureSelectionPreference {
    fn from(value: CliCapturePreference) -> Self {
        match value {
            CliCapturePreference::AutoHighFps => Self::AutoHighFps,
            CliCapturePreference::AutoLowLatency => Self::AutoLowLatency,
            CliCapturePreference::AutoBalanced => Self::AutoBalanced,
            CliCapturePreference::Manual => Self::Manual,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliProbeInputMode {
    Fixed,
    Latest,
}

impl From<CliProbeInputMode> for ModelProbeInputMode {
    fn from(value: CliProbeInputMode) -> Self {
        match value {
            CliProbeInputMode::Fixed => Self::Fixed,
            CliProbeInputMode::Latest => Self::Latest,
        }
    }
}

#[derive(Subcommand, Debug)]
enum ConfigCommand {
    /// Print the current persisted configuration.
    Show,
    /// Atomically persist one section field. The value accepts JSON or a plain string.
    Set {
        section: String,
        key: String,
        value: String,
        /// Reject the write unless the persisted revision matches this value.
        #[arg(long)]
        expected_revision: Option<u64>,
    },
}

#[derive(Serialize)]
#[serde(untagged)]
enum CommandOutput {
    Runtime(Box<RuntimeSnapshot>),
    Config(Box<AppConfig>),
    ConfigUpdate(Box<ConfigUpdate>),
    Model(ModelIngressResult),
    ModelRegistration(CatalogEngineRegistration),
    License(LicenseStatus),
    ModelProjects(Vec<ModelProject>),
    ModelVersions(Vec<ModelVersion>),
    ModelArtifacts(Vec<ModelArtifact>),
    ModelSwitch(ModelSwitchResponse),
    Executor(ExecutorStatus),
    DeviceButtons(DeviceButtons),
    DiagnosticMove(DiagnosticMoveResponse),
    CaptureCapabilities(novasight_core::CaptureCapabilities),
    Preview(PreviewSnapshot),
    Crosshair(CrosshairSnapshot),
    Json(serde_json::Value),
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match execute(cli).await {
        Ok(output) => match serde_json::to_string_pretty(&output) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("snapshot_encode_failed: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("{}: {error}", error.code());
            ExitCode::FAILURE
        }
    }
}

async fn execute(cli: Cli) -> Result<CommandOutput, CliError> {
    let client = ControlClient::new(resolve_control_socket());
    match cli.command {
        Command::Status => client
            .status()
            .await
            .map(Box::new)
            .map(CommandOutput::Runtime),
        Command::Start => client
            .start()
            .await
            .map(Box::new)
            .map(CommandOutput::Runtime),
        Command::Stop => client
            .stop()
            .await
            .map(Box::new)
            .map(CommandOutput::Runtime),
        Command::Restart => client
            .restart()
            .await
            .map(Box::new)
            .map(CommandOutput::Runtime),
        Command::EmergencyStop => client
            .emergency_stop()
            .await
            .map(Box::new)
            .map(CommandOutput::Runtime),
        Command::Config {
            command: ConfigCommand::Show,
        } => client
            .config()
            .await
            .map(Box::new)
            .map(CommandOutput::Config),
        Command::Config {
            command:
                ConfigCommand::Set {
                    section,
                    key,
                    value,
                    expected_revision,
                },
        } => {
            let value = serde_json::from_str(&value).unwrap_or(serde_json::Value::String(value));
            client
                .update_config_field(section, key, value, expected_revision)
                .await
                .map(Box::new)
                .map(CommandOutput::ConfigUpdate)
        }
        Command::License {
            command: LicenseCommand::Status,
        } => client.license_status().await.map(CommandOutput::License),
        Command::License {
            command: LicenseCommand::Activate { key_file },
        } => {
            let key =
                std::fs::read_to_string(&key_file).map_err(|source| CliError::ReadLicenseKey {
                    path: key_file,
                    source,
                })?;
            client
                .activate_license(key.trim())
                .await
                .map(CommandOutput::License)
        }
        Command::License {
            command: LicenseCommand::Clear,
        } => client.clear_license().await.map(CommandOutput::License),
        Command::Model {
            command: ModelCommand::Register { relative_path },
        } => client
            .register_catalog_engine(&relative_path)
            .await
            .map(CommandOutput::ModelRegistration),
        Command::Model {
            command: ModelCommand::Projects,
        } => client
            .model_projects()
            .await
            .map(CommandOutput::ModelProjects),
        Command::Model {
            command: ModelCommand::Versions { project_id },
        } => client
            .model_versions(project_id)
            .await
            .map(CommandOutput::ModelVersions),
        Command::Model {
            command: ModelCommand::Artifacts { version_id },
        } => client
            .model_artifacts(version_id)
            .await
            .map(CommandOutput::ModelArtifacts),
        Command::Model {
            command: ModelCommand::Inspect { artifact_id },
        } => client
            .inspect_model(artifact_id)
            .await
            .map(CommandOutput::Model),
        Command::Model {
            command: ModelCommand::Profile { artifact_id },
        } => client
            .model_profile(artifact_id)
            .await
            .map(CommandOutput::Model),
        Command::Model {
            command: ModelCommand::Recommend { artifact_id },
        } => client
            .deepstream_recommendation(artifact_id)
            .await
            .map(CommandOutput::Json),
        Command::Model {
            command:
                ModelCommand::Configure {
                    artifact_id,
                    request,
                },
        } => {
            let bytes = std::fs::read(&request).map_err(|source| CliError::ReadProfileRequest {
                path: request.clone(),
                source,
            })?;
            let profile = serde_json::from_slice::<ModelProfileConfigureRequest>(&bytes)
                .map_err(CliError::DecodeProfileRequest)?;
            client
                .configure_model(artifact_id, &profile)
                .await
                .map(CommandOutput::Model)
        }
        Command::Model {
            command:
                ModelCommand::Probe {
                    artifact_id,
                    input_mode,
                },
        } => client
            .probe_model(artifact_id, input_mode.into())
            .await
            .map(CommandOutput::Model),
        Command::Model {
            command:
                ModelCommand::Publish {
                    project_id,
                    artifact_id,
                    parser_preset,
                },
        } => client
            .publish_model(project_id, artifact_id, &parser_preset)
            .await
            .map(CommandOutput::ModelSwitch),
        Command::Model {
            command: ModelCommand::Rollback { project_id },
        } => client
            .rollback_model(project_id)
            .await
            .map(CommandOutput::ModelSwitch),
        Command::Device {
            command: DeviceCommand::Status,
        } => client.executor_status().await.map(CommandOutput::Executor),
        Command::Device {
            command: DeviceCommand::Buttons,
        } => client
            .device_buttons()
            .await
            .map(CommandOutput::DeviceButtons),
        Command::Device {
            command: DeviceCommand::Move { dx, dy },
        } => client
            .diagnostic_move(dx, dy)
            .await
            .map(CommandOutput::DiagnosticMove),
        Command::Capture {
            command: CaptureCommand::Status,
        } => client.capture_state().await.map(CommandOutput::Json),
        Command::Capture {
            command: CaptureCommand::Capabilities { device },
        } => client
            .capture_capabilities(&device)
            .await
            .map(CommandOutput::CaptureCapabilities),
        Command::Capture {
            command:
                CaptureCommand::Select {
                    device,
                    preference,
                    pixel_format,
                    width,
                    height,
                    fps,
                },
        } => {
            let manual = match (pixel_format.as_deref(), width, height, fps) {
                (Some(format), Some(width), Some(height), Some(fps)) => {
                    Some((format, width, height, fps))
                }
                _ => None,
            };
            if (preference == CliCapturePreference::Manual) != manual.is_some() {
                return Err(CliError::InvalidCaptureSelection);
            }
            client
                .select_capture(&device, preference.into(), manual)
                .await
                .map(CommandOutput::Json)
        }
        Command::Preview {
            command: PreviewCommand::Status,
        } => client.preview_status().await.map(CommandOutput::Preview),
        Command::Preview {
            command: PreviewCommand::On,
        } => client
            .set_preview_active(true)
            .await
            .map(CommandOutput::Preview),
        Command::Crosshair {
            command: CrosshairCommand::Status,
        } => client
            .crosshair_status()
            .await
            .map(CommandOutput::Crosshair),
        Command::Crosshair {
            command: CrosshairCommand::Learn,
        } => client.learn_crosshair().await.map(CommandOutput::Json),
        Command::Crosshair {
            command: CrosshairCommand::Clear,
        } => client.clear_crosshair().await.map(CommandOutput::Crosshair),
        Command::Preview {
            command: PreviewCommand::Off,
        } => client
            .set_preview_active(false)
            .await
            .map(CommandOutput::Preview),
    }
    .map_err(CliError::Client)
}

fn resolve_control_socket() -> PathBuf {
    if let Some(socket) = std::env::var_os(CONTROL_SOCKET_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    {
        return socket;
    }
    portable_control_socket()
}

fn portable_control_socket() -> PathBuf {
    let executable_root = std::env::current_exe()
        .ok()
        .and_then(|executable| executable.parent().map(PathBuf::from))
        .map(|executable_dir| {
            if executable_dir.file_name().is_some_and(|name| name == "bin") {
                executable_dir
                    .parent()
                    .map(PathBuf::from)
                    .unwrap_or(executable_dir)
            } else {
                executable_dir
            }
        });
    let executable_socket = executable_root
        .as_ref()
        .map(|root| root.join(PORTABLE_CONTROL_SOCKET));
    if let Some(socket) = executable_socket.as_ref().filter(|socket| socket.exists()) {
        return socket.to_owned();
    }
    if let Ok(current_dir) = std::env::current_dir() {
        let current_socket = current_dir.join(PORTABLE_CONTROL_SOCKET);
        if current_socket.exists() {
            return current_socket;
        }
    }
    if let Some(socket) = executable_socket {
        socket
    } else {
        PathBuf::from(PORTABLE_CONTROL_SOCKET)
    }
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Client(#[from] ClientError),
    #[error(
        "manual capture selection requires all of --pixel-format, --width, --height, and --fps; automatic preferences accept none of them"
    )]
    InvalidCaptureSelection,
    #[error("failed to read model profile request {}: {source}", path.display())]
    ReadProfileRequest {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to decode model profile request: {0}")]
    DecodeProfileRequest(#[source] serde_json::Error),
    #[error("failed to read license key {}: {source}", path.display())]
    ReadLicenseKey {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl CliError {
    const fn code(&self) -> &'static str {
        match self {
            Self::Client(error) => error.code(),
            Self::InvalidCaptureSelection => "CAPTURE_SELECTION_ARGUMENTS_INVALID",
            Self::ReadProfileRequest { .. } => "model_profile_request_read_failed",
            Self::DecodeProfileRequest(_) => "model_profile_request_invalid",
            Self::ReadLicenseKey { .. } => "license_key_read_failed",
        }
    }
}
