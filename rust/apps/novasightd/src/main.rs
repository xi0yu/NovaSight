//! `novasightd` — the long-running NovaSight daemon.
//!
//! Production startup fails closed until a real output adapter is
//! selected. Recording output is available only through `--dry-run`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use novasight_runtime::LoadedApplication;
use tracing_subscriber::EnvFilter;

mod server;

#[derive(Parser, Debug)]
#[command(name = "novasightd", about = "NovaSight runtime daemon")]
struct Args {
    /// Path to the external NovaSight YAML configuration.
    #[arg(long, default_value = "/etc/novasight/novasight.yaml")]
    config: PathBuf,

    /// Run preflight checks and exit; do not start the pipeline.
    #[arg(long)]
    check: bool,

    /// Use an in-memory recording output adapter; never sends hardware commands.
    #[arg(long)]
    dry_run: bool,
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
    let args = Args::parse();
    let loaded = match LoadedApplication::load(&args.config).await {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("{}: {error}", error.code());
            return ExitCode::FAILURE;
        }
    };

    if args.check {
        println!("PASS config readable path={}", args.config.display());
        return ExitCode::SUCCESS;
    }

    if !args.dry_run {
        if let Err(error) = loaded.config().require_production_adapters() {
            eprintln!("PRODUCTION_CONFIG_INVALID: {error}");
            return ExitCode::FAILURE;
        }
        eprintln!(
            "DEVICE_BACKEND_NOT_CONFIGURED: production output requires a real device adapter; use --dry-run only for diagnostics"
        );
        return ExitCode::FAILURE;
    }

    match server::run_dry_run(loaded).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", error.code());
            ExitCode::FAILURE
        }
    }
}
