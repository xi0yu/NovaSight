//! Production application composition for NovaSight.
//!
//! Production startup may remain explicitly uncommissioned; hardware output
//! stays closed until a real adapter has been provisioned.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
#[cfg(not(all(feature = "deepstream", target_os = "linux")))]
use std::sync::Arc;
#[cfg(not(all(feature = "deepstream", target_os = "linux")))]
use std::time::Instant;

use clap::Parser;
#[cfg(not(all(feature = "deepstream", target_os = "linux")))]
use novasight_core::{Clock, MonotonicNanos, PointerDevice, UncommissionedPointerDevice};
#[cfg(all(feature = "deepstream", target_os = "linux"))]
use novasight_runtime::NativeModelJobRunner;
#[cfg(not(all(feature = "deepstream", target_os = "linux")))]
use novasight_runtime::compose_pipeline_config;
use novasight_runtime::{ConfigService, LoadedApplication, RuntimeDependencies};
use novasight_store::config::AppConfig;
use novasight_store::model_catalog::SqliteModelCatalog;
use tracing_subscriber::EnvFilter;

#[cfg(all(feature = "deepstream", target_os = "linux"))]
use crate::pointer_adapter::configured_pointer_device_mode;
use crate::server;

#[cfg(all(feature = "deepstream", target_os = "linux"))]
use crate::live_perception;

const DEFAULT_CONFIG_PATH: &str = "data/novasight.yaml";
#[derive(Parser, Debug)]
#[command(about = "NovaSight runtime daemon")]
struct Args {
    /// Runtime configuration file for ordinary daemon startup.
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Run preflight checks and exit; do not start the pipeline.
    #[arg(long)]
    check: bool,
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
    let config_path = args
        .config
        .as_deref()
        .unwrap_or_else(|| Path::new(DEFAULT_CONFIG_PATH));
    let load = LoadedApplication::load_or_initialize_default(config_path).await;
    let loaded = match load {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("{}: {error}", error.code());
            return ExitCode::FAILURE;
        }
    };

    let configured_parser_library = loaded
        .config()
        .inference
        .clone()
        .unwrap_or_default()
        .deepstream_parser_library;
    let parser_library = resolve_deepstream_parser_library(
        &configured_parser_library,
        option_env!("NOVASIGHT_DEEPSTREAM_PARSER_LIBRARY").map(Path::new),
    );
    #[cfg(not(feature = "deepstream"))]
    let _ = &parser_library;
    let mode = daemon_mode();
    if args.check {
        #[cfg(all(feature = "deepstream", target_os = "linux"))]
        if let Err(error) = loaded.config().require_production_adapters() {
            eprintln!("PRODUCTION_CONFIG_INVALID: {error}");
            return ExitCode::FAILURE;
        }
        if let Err(error) = server::preflight_license_policy(mode) {
            eprintln!("{}: {error}", error.code());
            return ExitCode::FAILURE;
        }
        if let Err(error) = server::preflight_instance_lock(mode) {
            eprintln!("{}: {error}", error.code());
            return ExitCode::FAILURE;
        }
        #[cfg(all(feature = "deepstream", target_os = "linux"))]
        {
            let model_jobs = NativeModelJobRunner::new();
            if let Err(error) = model_jobs.preflight().await {
                eprintln!("MODEL_INGRESS_PREFLIGHT_FAILED: {error}");
                return ExitCode::FAILURE;
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
            if let Err(error) = live_perception::preflight_live_production(
                loaded.config(),
                &model_catalog,
                &parser_library,
            ) {
                eprintln!("PRODUCTION_PREFLIGHT_FAILED: {error}");
                return ExitCode::FAILURE;
            }
            let pointer_adapter =
                match configured_pointer_device_mode(loaded.config().device.as_ref()) {
                    novasight_core::PointerDeviceMode::Commissioned => "commissioned",
                    novasight_core::PointerDeviceMode::Uncommissioned => "uncommissioned",
                };
            println!(
                "PASS mode={} config={} configured_output_enabled={} daemon_transport=http1-unix license_verifier=ready instance_guard=ready model_ingress=native_tensorrt model_contract=ready deepstream_native_runtime=ready pipeline_constructed=true pointer_adapter={} capture_not_started=true pointer_not_connected=true",
                if cfg!(debug_assertions) {
                    "development_hardware"
                } else {
                    "production"
                },
                config_path.display(),
                loaded.config().control.output_enabled,
                pointer_adapter
            );
            return ExitCode::SUCCESS;
        }
        #[cfg(not(all(feature = "deepstream", target_os = "linux")))]
        {
            if let Err(error) = compose_pipeline_config(loaded.config(), None) {
                eprintln!("HOST_PREVIEW_CONFIG_INVALID: {error}");
                return ExitCode::FAILURE;
            }
            println!(
                "PASS mode={} config={} configured_output_enabled={} daemon_transport=http1-unix license_verifier=ready instance_guard=not_required model_ingress=unavailable perception_adapter=absent pointer_adapter=uncommissioned hardware_output_enabled=false",
                mode.label(),
                config_path.display(),
                loaded.config().control.output_enabled,
            );
            return ExitCode::SUCCESS;
        }
    }

    #[cfg(all(feature = "deepstream", target_os = "linux"))]
    if let Err(error) = loaded.config().require_production_adapters() {
        eprintln!("PRODUCTION_CONFIG_INVALID: {error}");
        return ExitCode::FAILURE;
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
    if let Err(error) = model_catalog.catalog(true) {
        eprintln!("MODEL_CATALOG_WARMUP_FAILED: {error}");
    }
    let config_service = ConfigService::new(loaded.config_path(), loaded.config().clone());
    let dependencies = match build_runtime_dependencies(
        loaded.config(),
        config_service.clone(),
        model_catalog.clone(),
        parser_library.clone(),
    ) {
        Ok(dependencies) => dependencies,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    #[cfg(all(feature = "deepstream", target_os = "linux"))]
    let dependencies = dependencies.with_model_jobs(NativeModelJobRunner::new());
    let dependencies = dependencies.with_output_enabled(
        mode.hardware_output_enabled() && loaded.config().control.output_enabled,
    );
    match server::run_daemon(loaded, dependencies, config_service, model_catalog, mode).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", error.code());
            ExitCode::FAILURE
        }
    }
}

#[cfg(all(feature = "deepstream", target_os = "linux"))]
const fn daemon_mode() -> server::DaemonMode {
    server::DaemonMode::Hardware
}

#[cfg(not(all(feature = "deepstream", target_os = "linux")))]
const fn daemon_mode() -> server::DaemonMode {
    server::DaemonMode::HostPreview
}

fn resolve_deepstream_parser_library(configured: &Path, bundled: Option<&Path>) -> PathBuf {
    if configured == Path::new("auto") {
        return bundled.unwrap_or(configured).to_owned();
    }
    configured.to_owned()
}

#[cfg(all(feature = "deepstream", target_os = "linux"))]
fn build_runtime_dependencies(
    config: &AppConfig,
    config_service: ConfigService,
    model_catalog: SqliteModelCatalog,
    parser_library: PathBuf,
) -> Result<RuntimeDependencies, String> {
    live_perception::build_live_production_dependencies(
        config,
        config_service,
        model_catalog,
        parser_library,
    )
    .map_err(|error| format!("PRODUCTION_RUNTIME_INVALID: {error}"))
}

#[cfg(not(all(feature = "deepstream", target_os = "linux")))]
fn build_runtime_dependencies(
    config: &AppConfig,
    _config_service: ConfigService,
    model_catalog: SqliteModelCatalog,
    _parser_library: PathBuf,
) -> Result<RuntimeDependencies, String> {
    let pipeline = compose_pipeline_config(config, None)
        .map_err(|error| format!("HOST_PREVIEW_RUNTIME_INVALID: {error}"))?;
    let clock: Arc<dyn Clock> = Arc::new(HostMonotonicClock::default());
    // Absence at the existing perception seam means no frame producer is
    // started. The uncommissioned device keeps every output path fail-closed.
    let device: Arc<dyn PointerDevice> = Arc::new(UncommissionedPointerDevice);
    Ok(RuntimeDependencies::new(clock, device, pipeline).with_model_catalog(model_catalog))
}

#[cfg(not(all(feature = "deepstream", target_os = "linux")))]
#[derive(Debug)]
struct HostMonotonicClock {
    origin: Instant,
}

#[cfg(not(all(feature = "deepstream", target_os = "linux")))]
impl Default for HostMonotonicClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

#[cfg(not(all(feature = "deepstream", target_os = "linux")))]
impl Clock for HostMonotonicClock {
    fn now(&self) -> MonotonicNanos {
        MonotonicNanos(self.origin.elapsed().as_nanos().min(u64::MAX as u128) as u64)
    }
}
