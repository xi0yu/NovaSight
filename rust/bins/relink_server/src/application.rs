//! Production application composition for NovaSight.
//!
//! Production startup fails closed until a real output adapter is
//! selected. Recording output is available only through `--dry-run`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use novasight_runtime::{
    ConfigService, LoadedApplication, OfflineModelJobRunner, RuntimeDependencies,
};
use novasight_store::model_catalog::SqliteModelCatalog;
use tracing_subscriber::EnvFilter;

use crate::server;

#[cfg(all(feature = "deepstream", target_os = "linux"))]
use crate::live_perception;

#[derive(Parser, Debug)]
#[command(about = "NovaSight runtime daemon")]
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

    /// Fixed Python executable used only for allowlisted offline model jobs.
    #[arg(long, default_value = "/usr/bin/python3")]
    model_job_python: PathBuf,

    /// Fixed helper script for inspect/configure/probe; API callers cannot override it.
    #[arg(long)]
    model_job_script: Option<PathBuf>,

    /// Root under which each allowlisted model helper gets a private working directory.
    #[arg(long, default_value = ".")]
    model_job_workdir: PathBuf,

    /// Hard timeout applied to every offline model helper process.
    #[arg(long, default_value_t = 60)]
    model_job_timeout_seconds: u64,
}

fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| "novasight=info".into()),
        )
        .try_init();
}

pub async fn entry() -> ExitCode {
    init_logging();
    let args = Args::parse();
    let loaded = match LoadedApplication::load(&args.config).await {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("{}: {error}", error.code());
            return ExitCode::FAILURE;
        }
    };

    let parser_library = loaded
        .config()
        .inference
        .clone()
        .unwrap_or_default()
        .deepstream_parser_library;
    let model_job_script = resolve_model_job_script(args.model_job_script.as_deref());
    let model_job_code_root = model_job_script
        .canonicalize()
        .ok()
        .and_then(|path| path.parent().and_then(Path::parent).map(Path::to_owned))
        .unwrap_or_else(|| args.model_job_workdir.clone())
        .join("novasight");
    let model_jobs = match OfflineModelJobRunner::new(
        &args.model_job_python,
        &model_job_script,
        &args.model_job_workdir,
        parser_library,
        Duration::from_secs(args.model_job_timeout_seconds),
        1024 * 1024,
    )
    .and_then(|runner| runner.pin_python_code(model_job_code_root))
    {
        Ok(runner) => runner,
        Err(error) => {
            eprintln!("MODEL_INGRESS_CONFIG_INVALID: {error}");
            return ExitCode::FAILURE;
        }
    };

    if args.check {
        if args.dry_run {
            if let Err(error) = loaded.config().validate_configured_adapters() {
                eprintln!("PREFLIGHT_CONFIG_INVALID: {error}");
                return ExitCode::FAILURE;
            }
            println!(
                "PASS mode=dry_run config={} model_ingress_helper=ready hardware_not_started=true",
                args.config.display()
            );
            return ExitCode::SUCCESS;
        }
        if let Err(error) = loaded.config().require_production_adapters() {
            eprintln!("PRODUCTION_CONFIG_INVALID: {error}");
            return ExitCode::FAILURE;
        }
        #[cfg(all(feature = "deepstream", target_os = "linux"))]
        {
            let model_catalog = match SqliteModelCatalog::open_with_model_root(
                &loaded.config().paths.database,
                &loaded.config().paths.model_dir,
            ) {
                Ok(catalog) => catalog,
                Err(error) => {
                    eprintln!("MODEL_CATALOG_OPEN_FAILED: {error}");
                    return ExitCode::FAILURE;
                }
            };
            if let Err(error) =
                live_perception::preflight_live_production(loaded.config(), &model_catalog)
            {
                eprintln!("PRODUCTION_PREFLIGHT_FAILED: {error}");
                return ExitCode::FAILURE;
            }
            println!(
                "PASS mode=production config={} model_ingress_helper=ready model_contract=ready deepstream_native_runtime=ready pipeline_constructed=true capture_not_started=true pointer_not_connected=true",
                args.config.display()
            );
            return ExitCode::SUCCESS;
        }
        #[cfg(not(all(feature = "deepstream", target_os = "linux")))]
        {
            eprintln!(
                "PRODUCTION_RUNTIME_UNAVAILABLE: rebuild novasightd on Linux with --features deepstream"
            );
            return ExitCode::FAILURE;
        }
    }

    let model_catalog = match SqliteModelCatalog::open_with_model_root(
        &loaded.config().paths.database,
        &loaded.config().paths.model_dir,
    ) {
        Ok(catalog) => catalog,
        Err(error) => {
            eprintln!("MODEL_CATALOG_OPEN_FAILED: {error}");
            return ExitCode::FAILURE;
        }
    };
    let config_service = ConfigService::new(loaded.config_path(), loaded.config().clone());

    let (dependencies, mode) = if args.dry_run && args.live_perception {
        #[cfg(all(feature = "deepstream", target_os = "linux"))]
        {
            match live_perception::build_live_recording_dependencies(
                loaded.config(),
                config_service.clone(),
                model_catalog.clone(),
            ) {
                Ok(dependencies) => (dependencies, server::DaemonMode::DryRun),
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
    } else if args.dry_run {
        (
            RuntimeDependencies::recording().with_model_catalog(model_catalog.clone()),
            server::DaemonMode::DryRun,
        )
    } else {
        if let Err(error) = loaded.config().require_production_adapters() {
            eprintln!("PRODUCTION_CONFIG_INVALID: {error}");
            return ExitCode::FAILURE;
        }
        #[cfg(all(feature = "deepstream", target_os = "linux"))]
        {
            match live_perception::build_live_production_dependencies(
                loaded.config(),
                config_service.clone(),
                model_catalog.clone(),
            ) {
                Ok(dependencies) => (dependencies, server::DaemonMode::Production),
                Err(error) => {
                    eprintln!("PRODUCTION_RUNTIME_INVALID: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        #[cfg(not(all(feature = "deepstream", target_os = "linux")))]
        {
            eprintln!(
                "PRODUCTION_RUNTIME_UNAVAILABLE: rebuild novasightd on Linux with --features deepstream"
            );
            return ExitCode::FAILURE;
        }
    };

    let dependencies = dependencies.with_model_jobs(model_jobs);
    match server::run_daemon(loaded, dependencies, config_service, model_catalog, mode).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", error.code());
            ExitCode::FAILURE
        }
    }
}

fn resolve_model_job_script(configured: Option<&Path>) -> PathBuf {
    if let Some(path) = configured {
        return path.to_owned();
    }
    let working_tree_path = PathBuf::from("scripts/model_ingress_job.py");
    if working_tree_path.is_file() {
        return working_tree_path;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/model_ingress_job.py")
}
