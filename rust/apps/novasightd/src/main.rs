//! `novasightd` — the long-running NovaSight daemon.
//!
//! Commit 1 ships an empty entrypoint that initializes logging and
//! returns success. The Application bootstrap lands in Commit 4.

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "novasightd", about = "NovaSight runtime daemon")]
struct Args {
    /// Path to the external NovaSight TOML configuration.
    #[arg(long, default_value = "/etc/novasight/novasight.toml")]
    config: String,

    /// Run preflight checks and exit; do not start the pipeline.
    #[arg(long)]
    check: bool,

    /// Stay in the foreground; do not daemonize.
    #[arg(long)]
    foreground: bool,
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
    let _args = Args::parse();
    Ok(())
}
