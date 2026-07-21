//! `novasightctl` — the NovaSight command-line client.
//!
//! Commit 1 ships an empty entrypoint that initializes logging and
//! returns success. The real CLI surface lands in Commit 6.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "novasightctl", about = "NovaSight daemon command-line client")]
struct Cli {
    #[arg(long, default_value = "/run/novasight/novasightd.sock")]
    socket: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Status,
    Start,
    Stop,
    Restart,
    EmergencyStop,
    Diagnose,
}

fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| "novasight=info".into()),
        )
        .try_init();
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    init_logging();
    let _cli = Cli::parse();
    Ok(())
}
