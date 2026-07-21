use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

mod bootstrap;
mod shutdown;

#[derive(Debug, Parser)]
#[command(name = "relink_server", about = "NovaSight replay backend")]
struct Cli {
    /// Path to the external NovaSight YAML configuration.
    #[arg(long, default_value = "config/novasight.yaml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> ExitCode {
    let _ = tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .try_init();

    match bootstrap::run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", error.code());
            ExitCode::FAILURE
        }
    }
}
