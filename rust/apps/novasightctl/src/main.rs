//! `novasightctl` — the thin local client for the daemon authority.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use novasight_client::{ClientError, ControlClient};
use novasight_runtime::{AppConfig, ConfigUpdate, RuntimeSnapshot};
use serde::Serialize;
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

async fn execute(cli: Cli) -> Result<CommandOutput, ClientError> {
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
    }
}
