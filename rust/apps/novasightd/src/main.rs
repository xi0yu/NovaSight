//! `novasightd` — the long-running NovaSight daemon.
//!
//! Production startup fails closed until a real output adapter is
//! selected. Recording output is available only through `--dry-run`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use novasight_runtime::{LoadedApplication, RuntimeDependencies};
use tracing_subscriber::EnvFilter;

mod server;

#[cfg(all(feature = "deepstream", target_os = "linux"))]
mod live_perception;

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

    /// Run the real Jetson DeepStream perception chain while keeping hardware
    /// output on the in-memory recording adapter.
    #[arg(long, requires = "dry_run")]
    live_perception: bool,
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

    let dependencies = if args.live_perception {
        #[cfg(all(feature = "deepstream", target_os = "linux"))]
        {
            match live_perception::build_live_recording_dependencies(loaded.config()) {
                Ok(dependencies) => dependencies,
                Err(error) => {
                    eprintln!("LIVE_PERCEPTION_CONFIG_INVALID: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        #[cfg(not(all(feature = "deepstream", target_os = "linux")))]
        {
            eprintln!(
                "LIVE_PERCEPTION_UNAVAILABLE: rebuild novasightd on Linux with --features deepstream"
            );
            return ExitCode::FAILURE;
        }
    } else {
        RuntimeDependencies::recording()
    };

    match server::run_dry_run(loaded, dependencies).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", error.code());
            ExitCode::FAILURE
        }
    }
}
