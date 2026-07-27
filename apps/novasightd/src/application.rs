//! Production application composition for NovaSight.
//!
//! Production startup may remain explicitly uncommissioned; hardware output
//! stays closed until a real adapter has been provisioned. Recording output is
//! available only through `--dry-run`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use novasight_core::controller::recoil::RecoilConfig;
#[cfg(feature = "deepstream")]
use novasight_runtime::NativeModelJobRunner;
use novasight_runtime::{ConfigService, LoadedApplication, RuntimeDependencies};
use novasight_store::model_catalog::SqliteModelCatalog;
use tracing_subscriber::EnvFilter;

#[cfg(all(feature = "deepstream", target_os = "linux"))]
use crate::pointer_adapter::configured_pointer_device_mode;
use crate::server;

#[cfg(all(feature = "deepstream", target_os = "linux"))]
use crate::live_perception;

#[derive(Parser, Debug)]
#[command(about = "NovaSight runtime daemon")]
struct Args {
    /// Path to the external NovaSight YAML configuration.
    #[arg(long, default_value = ".config/novasight.yaml")]
    config: PathBuf,

    /// Run preflight checks and exit; do not start the pipeline.
    #[arg(long)]
    check: bool,

    /// Isolated development mode with recording adapters; never opens the
    /// production DeepStream or physical pointer-device path.
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
    if args.check {
        if args.dry_run {
            if let Err(error) = loaded.config().validate_configured_adapters() {
                eprintln!("PREFLIGHT_CONFIG_INVALID: {error}");
                return ExitCode::FAILURE;
            }
            if let Err(error) = server::preflight_license_policy(server::DaemonMode::DryRun) {
                eprintln!("{}: {error}", error.code());
                return ExitCode::FAILURE;
            }
            if let Err(error) = server::preflight_instance_lock(server::DaemonMode::DryRun) {
                eprintln!("{}: {error}", error.code());
                return ExitCode::FAILURE;
            }
            println!(
                "PASS mode=dry_run config={} configured_output_enabled={} model_ingress=not_started hardware_not_started=true",
                args.config.display(),
                loaded.config().control.output_enabled
            );
            return ExitCode::SUCCESS;
        }
        if let Err(error) = loaded.config().require_production_adapters() {
            eprintln!("PRODUCTION_CONFIG_INVALID: {error}");
            return ExitCode::FAILURE;
        }
        if let Err(error) = server::preflight_license_policy(server::DaemonMode::Hardware) {
            eprintln!("{}: {error}", error.code());
            return ExitCode::FAILURE;
        }
        if let Err(error) = server::preflight_instance_lock(server::DaemonMode::Hardware) {
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
                "PASS mode={} config={} configured_output_enabled={} license_verifier=ready instance_guard=ready model_ingress=native_tensorrt model_contract=ready deepstream_native_runtime=ready pipeline_constructed=true pointer_adapter={} capture_not_started=true pointer_not_connected=true",
                if cfg!(debug_assertions) {
                    "development_hardware"
                } else {
                    "production"
                },
                args.config.display(),
                loaded.config().control.output_enabled,
                pointer_adapter
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

    if !args.dry_run
        && let Err(error) = loaded.config().require_production_adapters()
    {
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
    let config_service = ConfigService::new(loaded.config_path(), loaded.config().clone());
    let (dependencies, mode) = if args.dry_run {
        (
            RuntimeDependencies::recording().with_model_catalog(model_catalog.clone()),
            server::DaemonMode::DryRun,
        )
    } else {
        #[cfg(all(feature = "deepstream", target_os = "linux"))]
        {
            match live_perception::build_live_production_dependencies(
                loaded.config(),
                config_service.clone(),
                model_catalog.clone(),
                parser_library.clone(),
            ) {
                Ok(dependencies) => (dependencies, server::DaemonMode::Hardware),
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

    #[cfg(feature = "deepstream")]
    let dependencies = dependencies.with_model_jobs(NativeModelJobRunner::new());
    let dependencies = dependencies
        .with_output_enabled(loaded.config().control.output_enabled)
        .with_recoil(RecoilConfig {
            enabled: loaded.config().control.recoil.enabled,
            base_rate_counts_s: loaded.config().control.recoil.base_rate_counts_s,
            max_rate_counts_s: loaded.config().control.recoil.max_rate_counts_s,
            startup_ms: loaded.config().control.recoil.startup_ms,
            positive_deadzone_norm: loaded.config().control.recoil.positive_deadzone_norm,
            negative_deadzone_norm: loaded.config().control.recoil.negative_deadzone_norm,
            full_brake_error_norm: loaded.config().control.recoil.full_brake_error_norm,
            fast_add_gain_counts_s: loaded.config().control.recoil.fast_add_gain_counts_s,
            max_fast_add_ratio: loaded.config().control.recoil.max_fast_add_ratio,
            stale_threshold_ms: loaded.config().control.recoil.stale_threshold_ms,
        });
    match server::run_daemon(loaded, dependencies, config_service, model_catalog, mode).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", error.code());
            ExitCode::FAILURE
        }
    }
}

fn resolve_deepstream_parser_library(configured: &Path, bundled: Option<&Path>) -> PathBuf {
    if configured == Path::new("auto") {
        return bundled.unwrap_or(configured).to_owned();
    }
    configured.to_owned()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::resolve_deepstream_parser_library;

    #[test]
    fn cargo_managed_parser_replaces_the_auto_sentinel() {
        let bundled = Path::new("/cargo/out/libnovasight_parser.so");

        assert_eq!(
            resolve_deepstream_parser_library(Path::new("auto"), Some(bundled)),
            bundled
        );
    }

    #[test]
    fn explicitly_configured_parser_path_is_preserved() {
        let configured = Path::new("/opt/novasight/lib/custom-parser.so");

        assert_eq!(
            resolve_deepstream_parser_library(
                configured,
                Some(Path::new("/cargo/out/libnovasight_parser.so")),
            ),
            configured
        );
    }
}
