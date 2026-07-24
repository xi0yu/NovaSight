//! Production application composition for NovaSight.
//!
//! Production startup may remain explicitly uncommissioned; hardware output
//! stays closed until a real adapter has been provisioned. Recording output is
//! available only through `--dry-run`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use novasight_core::control::humanized_motion::MotionRuntimeParameters;
use novasight_core::control::recoil::RecoilConfig;
use novasight_pipeline::MotionProfileHub;
use novasight_runtime::{
    ConfigService, LoadedApplication, OfflineModelJobRunner, RuntimeDependencies,
};
use novasight_store::model_catalog::SqliteModelCatalog;
use novasight_store::motion_profile::MotionProfileRepository;
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
    let model_job_script = resolve_model_job_script(args.model_job_script.as_deref());
    let python_package_root = model_job_script
        .canonicalize()
        .ok()
        .and_then(|path| path.parent().and_then(Path::parent).map(Path::to_owned))
        .unwrap_or_else(|| args.model_job_workdir.clone());
    let model_job_code_root = python_package_root.join("novasight");
    let model_jobs = match OfflineModelJobRunner::new(
        &args.model_job_python,
        &model_job_script,
        &args.model_job_workdir,
        parser_library.clone(),
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

    let motion_repository =
        match MotionProfileRepository::open(loaded.config().paths.data_dir.join("motion")) {
            Ok(repository) => repository,
            Err(error) => {
                eprintln!("MOTION_PROFILE_OPEN_FAILED: {error}");
                return ExitCode::FAILURE;
            }
        };
    let motion_tuning = &loaded.config().control.humanized_motion;
    let motion_hub = MotionProfileHub::new(
        motion_tuning.builtin_fitts_a_ms,
        motion_tuning.builtin_fitts_b_ms,
        motion_tuning.builtin_side_ratio,
        MotionRuntimeParameters {
            spatial_curve_enabled: motion_tuning.spatial_curve_enabled,
            side_scale: motion_tuning.side_scale,
            max_side_ratio: motion_tuning.max_side_ratio,
            near_fade_start_px: motion_tuning.near_fade_start_px,
            micro_bypass_px: motion_tuning.micro_bypass_px,
            dynamic_rebase_ratio: motion_tuning.dynamic_rebase_ratio,
            minimum_jerk_fallback: motion_tuning.minimum_jerk_fallback,
            terminal_feedback_gain: motion_tuning.terminal_feedback_gain,
            ..MotionRuntimeParameters::default()
        },
    );
    if motion_tuning.enabled {
        if motion_tuning.active_profile.is_empty() || motion_tuning.active_profile == "builtin" {
            motion_hub.activate_builtin_startup();
        } else {
            let profile = match motion_repository.profile(&motion_tuning.active_profile) {
                Ok(profile) => profile,
                Err(error) => {
                    eprintln!(
                        "MOTION_PROFILE_LOAD_FAILED: profile {}: {error}",
                        motion_tuning.active_profile
                    );
                    return ExitCode::FAILURE;
                }
            };
            if let Err(error) = motion_hub.activate_startup(Some(profile)) {
                eprintln!(
                    "MOTION_PROFILE_INVALID: profile {}: {error}",
                    motion_tuning.active_profile
                );
                return ExitCode::FAILURE;
            }
        }
    }

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
            if let Err(error) = model_jobs.preflight().await {
                eprintln!("MODEL_INGRESS_PREFLIGHT_FAILED: {error}");
                return ExitCode::FAILURE;
            }
            println!(
                "PASS mode=dry_run config={} configured_output_enabled={} model_ingress_helper=ready motion_profile=ready hardware_not_started=true",
                args.config.display(),
                loaded.config().control.output_enabled
            );
            return ExitCode::SUCCESS;
        }
        if let Err(error) = loaded.config().require_production_adapters() {
            eprintln!("PRODUCTION_CONFIG_INVALID: {error}");
            return ExitCode::FAILURE;
        }
        if let Err(error) = server::preflight_license_policy(server::DaemonMode::Production) {
            eprintln!("{}: {error}", error.code());
            return ExitCode::FAILURE;
        }
        if let Err(error) = server::preflight_instance_lock(server::DaemonMode::Production) {
            eprintln!("{}: {error}", error.code());
            return ExitCode::FAILURE;
        }
        if let Err(error) = model_jobs.preflight().await {
            eprintln!("MODEL_INGRESS_PREFLIGHT_FAILED: {error}");
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
            if let Err(error) = live_perception::preflight_live_production(
                loaded.config(),
                &model_catalog,
                &python_package_root,
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
                "PASS mode=production config={} configured_output_enabled={} license_verifier=ready instance_lock=ready model_ingress_helper=ready motion_profile=ready model_contract=ready deepstream_native_runtime=ready pipeline_constructed=true pointer_adapter={} capture_not_started=true pointer_not_connected=true",
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
    let config_service = ConfigService::new(loaded.config_path(), loaded.config().clone());
    let (dependencies, mode) = if args.dry_run && args.live_perception {
        #[cfg(all(feature = "deepstream", target_os = "linux"))]
        {
            match live_perception::build_live_recording_dependencies(
                loaded.config(),
                config_service.clone(),
                model_catalog.clone(),
                parser_library.clone(),
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
        #[cfg(all(feature = "deepstream", target_os = "linux"))]
        {
            match live_perception::build_live_production_dependencies(
                loaded.config(),
                config_service.clone(),
                model_catalog.clone(),
                &python_package_root,
                parser_library.clone(),
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

    let dependencies = dependencies
        .with_model_jobs(model_jobs)
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
        })
        .with_motion_profiles(motion_hub, motion_repository);
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
    if let Ok(executable) = std::env::current_exe()
        && let Some(path) = bundled_model_job_script(&executable)
    {
        return path;
    }
    let working_tree_path = PathBuf::from("scripts/model_ingress_job.py");
    if working_tree_path.is_file() {
        return working_tree_path;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/model_ingress_job.py")
}

fn resolve_deepstream_parser_library(configured: &Path, bundled: Option<&Path>) -> PathBuf {
    if configured == Path::new("auto") {
        return bundled.unwrap_or(configured).to_owned();
    }
    configured.to_owned()
}

fn bundled_model_job_script(executable: &Path) -> Option<PathBuf> {
    let release_root = executable.parent()?.parent()?;
    let candidate = release_root.join("scripts/model_ingress_job.py");
    candidate.is_file().then_some(candidate)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::{bundled_model_job_script, resolve_deepstream_parser_library};

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

    #[test]
    fn installed_daemon_discovers_worker_in_its_own_release() {
        let root =
            std::env::temp_dir().join(format!("novasight-bundled-worker-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("bin")).unwrap();
        fs::create_dir_all(root.join("scripts")).unwrap();
        fs::write(root.join("scripts/model_ingress_job.py"), b"# worker\n").unwrap();

        assert_eq!(
            bundled_model_job_script(&root.join("bin/novasightd")),
            Some(root.join("scripts/model_ingress_job.py"))
        );

        fs::remove_dir_all(root).unwrap();
    }
}
