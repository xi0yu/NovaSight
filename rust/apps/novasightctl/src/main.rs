//! `novasightctl` — the thin local client for the daemon authority.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use novasight_client::{ClientError, ControlClient};
use novasight_runtime::{
    AppConfig, ConfigUpdate, ModelIngressResult, ModelProbeInputMode, ModelProfileConfigureRequest,
    RuntimeSnapshot,
};
use serde::Serialize;
use thiserror::Error;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "novasightctl", about = "NovaSight daemon command-line client")]
struct Cli {
    /// Unix domain socket exposed by novasightd.
    #[arg(long, default_value = "/run/novasight/novasightd.sock")]
    socket: PathBuf,

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
    /// Inspect, configure, and diagnostically execute TensorRT model artifacts.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCommand {
    /// Deserialize an Engine and write its initial immutable profile.
    Inspect { artifact_id: i64 },
    /// Read the current profile from the unified Engine manifest.
    Profile { artifact_id: i64 },
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
    Runtime(RuntimeSnapshot),
    Config(AppConfig),
    ConfigUpdate(ConfigUpdate),
    Model(ModelIngressResult),
}

fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| "novasight=info".into()),
        )
        .try_init();
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    init_logging();
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
    let client = ControlClient::new(cli.socket);
    match cli.command {
        Command::Status => client.status().await.map(CommandOutput::Runtime),
        Command::Start => client.start().await.map(CommandOutput::Runtime),
        Command::Stop => client.stop().await.map(CommandOutput::Runtime),
        Command::Restart => client.restart().await.map(CommandOutput::Runtime),
        Command::EmergencyStop => client.emergency_stop().await.map(CommandOutput::Runtime),
        Command::Config {
            command: ConfigCommand::Show,
        } => client.config().await.map(CommandOutput::Config),
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
                .map(CommandOutput::ConfigUpdate)
        }
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
    }
    .map_err(CliError::Client)
}

#[derive(Debug, Error)]
enum CliError {
    #[error(transparent)]
    Client(#[from] ClientError),
    #[error("failed to read model profile request {}: {source}", path.display())]
    ReadProfileRequest {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to decode model profile request: {0}")]
    DecodeProfileRequest(#[source] serde_json::Error),
}

impl CliError {
    const fn code(&self) -> &'static str {
        match self {
            Self::Client(error) => error.code(),
            Self::ReadProfileRequest { .. } => "model_profile_request_read_failed",
            Self::DecodeProfileRequest(_) => "model_profile_request_invalid",
        }
    }
}
