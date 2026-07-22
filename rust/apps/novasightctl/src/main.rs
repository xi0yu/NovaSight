//! `novasightctl` — the thin local client for the daemon authority.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use novasight_client::{ClientError, ControlClient};
use novasight_runtime::RuntimeSnapshot;
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
        Ok(snapshot) => match serde_json::to_string_pretty(&snapshot) {
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

async fn execute(cli: Cli) -> Result<RuntimeSnapshot, ClientError> {
    let client = ControlClient::new(cli.socket);
    match cli.command {
        Command::Status => client.status().await,
        Command::Start => client.start().await,
        Command::Stop => client.stop().await,
        Command::Restart => client.restart().await,
        Command::EmergencyStop => client.emergency_stop().await,
    }
}
