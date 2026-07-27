//! Jetson production adapter composition.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use novasight_core::controller::DualPhaseConfig;
use novasight_core::tracking::TargetingConfig;
use novasight_core::{
    CaptureCapabilityProbe, CaptureSelectionPreference, Clock,
    MAX_DETECTIONS as DEEPSTREAM_MAX_DETECTIONS, PointerDevice, RuntimeEpoch,
    select_capture_profile_for_formats,
};
use novasight_pipeline::{
    CrosshairConfig as PipelineCrosshairConfig, CrosshairHub, ModelCandidate,
    ParserContract as PerceptionParserContract, PerceptionAdapter, PerceptionError,
    PerceptionEvent, PerceptionModelContract, PerceptionRuntimeContract, PerceptionSession,
    PipelineConfig, PipelineIngress, PreviewHub, TriggerMode as PipelineTriggerMode,
    validate_parser_preset,
};
use novasight_platform_jetson::SystemMonotonicClock;
use novasight_platform_jetson::deepstream::{
    CaptureFormat, CaptureProfile, CrosshairPipelineConfig, DeepStreamAdapter,
    DeepStreamPipelineSpec, DeepStreamSessionConfig, InferenceStage, LatestFrameExchange,
    ModelInput, PreviewPipelineConfig, Roi, SessionError, preflight_deepstream_runtime,
};
use novasight_platform_jetson::kmnet::{KmNetNativeConfig, KmNetNativeDevice, KmNetNativeError};
use novasight_platform_jetson::v4l2::V4l2CapabilityProbe;
use novasight_runtime::{ConfigService, RuntimeDependencies};
use novasight_store::config::{
    AppConfig, CaptureConfig, CapturePreference, ConfigValidationError, DeviceBackend,
    DeviceConfig, parse_target_class_aim_y_ratios, parse_target_class_filter,
    parse_target_class_priority,
};
use novasight_store::model_catalog::{RuntimeModelArtifact, SqliteModelCatalog};
use novasight_store::model_manifest::ModelManifest;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::pointer_adapter::select_uncommissioned_pointer_adapter;

pub(super) fn build_live_production_dependencies(
    config: &AppConfig,
    config_service: ConfigService,
    model_catalog: SqliteModelCatalog,
    parser_library: PathBuf,
) -> Result<RuntimeDependencies, LivePerceptionError> {
    let adapters = config
        .require_production_adapters()
        .map_err(LivePerceptionError::Config)?;
    if let Some(selected) = select_uncommissioned_pointer_adapter(adapters.device) {
        return build_live_dependencies(
            config,
            config_service,
            model_catalog,
            selected.device,
            selected.trigger_poll_interval_ms,
            parser_library,
        );
    }
    let device: Arc<dyn PointerDevice> = match adapters.device.backend {
        DeviceBackend::NativeUdp => Arc::new(build_native_kmnet(adapters.device)?),
    };
    build_live_dependencies(
        config,
        config_service,
        model_catalog,
        device,
        Some(adapters.device.trigger_poll_interval_ms),
        parser_library,
    )
}

/// Validate the production vision contract and linked native runtime without
/// opening capture, starting inference, or connecting the pointer device.
pub(super) fn preflight_live_production(
    config: &AppConfig,
    model_catalog: &SqliteModelCatalog,
    parser_library: &Path,
) -> Result<(), LivePerceptionError> {
    preflight_pointer_adapter(config)?;
    let preview = PreviewHub::new(config.consumers.preview);
    let crosshair = build_crosshair_hub(config)?;
    let session =
        build_deepstream_session_config(config, model_catalog, preview, crosshair, parser_library)?;
    preflight_deepstream_runtime(&session).map_err(LivePerceptionError::RuntimePreflight)
}

fn preflight_pointer_adapter(config: &AppConfig) -> Result<(), LivePerceptionError> {
    let adapters = config
        .require_production_adapters()
        .map_err(LivePerceptionError::Config)?;
    if select_uncommissioned_pointer_adapter(adapters.device).is_some() {
        return Ok(());
    }
    match adapters.device.backend {
        DeviceBackend::NativeUdp => build_native_kmnet(adapters.device).map(drop),
    }
}

fn build_native_kmnet(device: &DeviceConfig) -> Result<KmNetNativeDevice, LivePerceptionError> {
    let host = device
        .host
        .parse()
        .map_err(|_| LivePerceptionError::InvalidKmNetHost(device.host.clone()))?;
    KmNetNativeDevice::new(KmNetNativeConfig {
        host,
        port: device.port,
        uuid: device.uuid.clone(),
        monitor_port: device.monitor_port,
        connect_timeout: Duration::from_millis(device.connect_timeout_ms),
        request_timeout: Duration::from_millis(device.send_timeout_ms),
        monitor_timeout: Duration::from_millis(device.monitor_timeout_ms),
    })
    .map_err(LivePerceptionError::NativeKmNet)
}

fn build_live_dependencies(
    config: &AppConfig,
    config_service: ConfigService,
    model_catalog: SqliteModelCatalog,
    device: Arc<dyn PointerDevice>,
    trigger_poll_interval_ms: Option<u64>,
    parser_library: PathBuf,
) -> Result<RuntimeDependencies, LivePerceptionError> {
    let adapters = config
        .require_vision_adapters()
        .map_err(LivePerceptionError::Config)?;
    if !adapters.inference.enabled {
        return Err(LivePerceptionError::InferenceDisabled);
    }
    let clock: Arc<dyn Clock> = Arc::new(SystemMonotonicClock::default());
    let latest_frames = LatestFrameExchange::new();
    let preview = PreviewHub::new(adapters.consumers.preview);
    let crosshair = build_crosshair_hub(config)?;
    let mut dependencies = RuntimeDependencies::new(
        clock,
        device,
        PipelineConfig {
            targeting: TargetingConfig {
                target_fov_radius_px: adapters.pipeline.target_fov_radius_px,
                min_confidence: adapters.pipeline.target_min_confidence,
                track_max_age: adapters.pipeline.target_track_max_age,
                track_max_lost_age_ms: adapters.pipeline.target_track_max_lost_age_ms,
                tracker_max_match_distance: adapters.pipeline.tracker_max_match_distance,
                tracker_position_cost_weight: adapters.pipeline.tracker_position_cost_weight,
                tracker_iou_cost_weight: adapters.pipeline.tracker_iou_cost_weight,
                tracker_scale_cost_weight: adapters.pipeline.tracker_scale_cost_weight,
                tracker_max_size_ratio: adapters.pipeline.tracker_max_size_ratio,
                tracker_max_association_dt_ms: adapters.pipeline.tracker_max_association_dt_ms,
                kalman: Default::default(),
                class_priority: parse_target_class_priority(
                    &adapters.pipeline.target_class_priority,
                )
                .map_err(LivePerceptionError::Config)?,
                allowed_class_ids: parse_target_class_filter(
                    &adapters.pipeline.target_class_filter,
                )
                .map_err(LivePerceptionError::Config)?,
                selection_class_weight: adapters.pipeline.target_selection_class_weight,
                selection_distance_weight: adapters.pipeline.target_selection_distance_weight,
                sticky_bias: adapters.pipeline.target_sticky_bias,
                switch_min_preference_advantage: adapters
                    .pipeline
                    .target_switch_min_preference_advantage,
                switch_min_continuity_score: adapters.pipeline.target_switch_min_continuity_score,
                switch_delay_ms: adapters.pipeline.target_switch_delay_ms,
                aim_y_ratio: adapters.pipeline.target_aim_y_ratio,
                class_aim_y_ratios: parse_target_class_aim_y_ratios(
                    &adapters.pipeline.target_class_aim_y_ratios,
                )
                .map_err(LivePerceptionError::Config)?,
                candidate_max_aspect_ratio: adapters.pipeline.candidate_max_aspect_ratio,
            },
            control: DualPhaseConfig {
                freshness_threshold_ms: adapters.pipeline.freshness_threshold_ms,
                near_threshold_px: adapters.pipeline.near_threshold_px,
                projection_fov_x_deg: adapters.pipeline.projection_fov_x_deg,
                projection_counts_per_360: adapters.pipeline.projection_counts_per_360,
                projection_invert_y: adapters.pipeline.projection_invert_y,
                atan_scale_counts: adapters.pipeline.atan_scale_counts,
                far_kp: adapters.pipeline.far_kp,
                far_max_counts_per_update: adapters.pipeline.far_max_counts_per_update,
                near_kp: adapters.pipeline.near_kp,
                near_max_counts_per_update: adapters.pipeline.near_max_counts_per_update,
                arrival_radius_counts: adapters.pipeline.arrival_radius_counts,
                velocity_smoothing_frames: adapters.pipeline.velocity_smoothing_frames,
                velocity_history_reset_gap_ms: adapters.pipeline.velocity_history_reset_gap_ms,
                velocity_spread_base_px_ms: adapters.pipeline.velocity_spread_base_px_ms,
                velocity_spread_relative: adapters.pipeline.velocity_spread_relative,
                velocity_change_base_px_ms: adapters.pipeline.velocity_change_base_px_ms,
                velocity_change_relative: adapters.pipeline.velocity_change_relative,
                prediction_enabled: adapters.pipeline.prediction_enabled,
                prediction_lead_frames: adapters.pipeline.prediction_lead_frames,
                prediction_far_absolute_cap_px: adapters.pipeline.prediction_far_absolute_cap_px,
                prediction_far_base_cap_px: adapters.pipeline.prediction_far_base_cap_px,
                prediction_far_relative_cap: adapters.pipeline.prediction_far_relative_cap,
                prediction_near_absolute_cap_px: adapters.pipeline.prediction_near_absolute_cap_px,
                prediction_near_base_cap_px: adapters.pipeline.prediction_near_base_cap_px,
                prediction_near_relative_cap: adapters.pipeline.prediction_near_relative_cap,
                source_width: adapters.capture.width,
                roi_width: adapters.capture.roi_width,
                roi_height: adapters.capture.roi_height,
                observation_width: 0,
                observation_height: 0,
                residual_cap: adapters.pipeline.residual_cap,
            },
            max_command_age_ns: adapters.pipeline.max_command_age_ms * 1_000_000,
            output_interval_ms: adapters.pipeline.output_interval_ms,
            actuation_feedback_delay_ns: (adapters.pipeline.actuation_feedback_delay_ms
                * 1_000_000.0)
                .round() as u64,
            trigger_poll_interval_ms,
            trigger_mode: match config.control.trigger_mode {
                novasight_store::config::TriggerMode::Always => PipelineTriggerMode::Always,
                novasight_store::config::TriggerMode::Hardware => PipelineTriggerMode::Hardware,
            },
            ..PipelineConfig::default()
        },
    )
    .with_model_catalog(model_catalog.clone())
    .with_preview(preview.clone())
    .with_perception(Arc::new(CatalogDeepStreamAdapter {
        config: config_service,
        model_catalog,
        latest_frames,
        preview,
        crosshair: crosshair.clone(),
        parser_library,
    }));
    if let Some(crosshair) = crosshair {
        dependencies = dependencies.with_crosshair(crosshair);
    }
    Ok(dependencies)
}

fn resolve_capture_plan(capture: &CaptureConfig) -> Result<CaptureConfig, LivePerceptionError> {
    if capture.preference == CapturePreference::Manual {
        parse_capture_format(&capture.pixel_format)?;
        return Ok(capture.clone());
    }
    let capabilities = V4l2CapabilityProbe
        .probe(&capture.device.to_string_lossy())
        .map_err(|error| LivePerceptionError::CaptureProbe(error.to_string()))?;
    let preference = match capture.preference {
        CapturePreference::AutoHighFps => CaptureSelectionPreference::AutoHighFps,
        CapturePreference::AutoLowLatency => CaptureSelectionPreference::AutoLowLatency,
        CapturePreference::AutoBalanced => CaptureSelectionPreference::AutoBalanced,
        CapturePreference::Manual => unreachable!(),
    };
    let selected = select_capture_profile_for_formats(
        &capabilities,
        preference,
        None,
        &["MJPG", "NV12", "YUYV"],
    )
    .map_err(|error| LivePerceptionError::CaptureSelection(error.to_string()))?;
    let mut resolved = capture.clone();
    resolved.preference = CapturePreference::Manual;
    resolved.pixel_format = selected.pixel_format;
    resolved.width = selected.width;
    resolved.height = selected.height;
    resolved.fps = selected.fps;
    if resolved.roi_width == 0 || resolved.roi_height == 0 {
        resolved.roi_left = 0;
        resolved.roi_top = 0;
        resolved.roi_width = resolved.width;
        resolved.roi_height = resolved.height;
    }
    resolved
        .validate_runtime_plan()
        .map_err(LivePerceptionError::Config)?;
    Ok(resolved)
}

#[derive(Clone, Debug)]
struct CatalogDeepStreamAdapter {
    config: ConfigService,
    model_catalog: SqliteModelCatalog,
    latest_frames: LatestFrameExchange,
    preview: PreviewHub,
    crosshair: Option<CrosshairHub>,
    parser_library: PathBuf,
}

impl PerceptionAdapter for CatalogDeepStreamAdapter {
    fn preflight(&self) -> Result<(), PerceptionError> {
        let config = self.config.blocking_effective_snapshot();
        build_deepstream_session_config(
            &config,
            &self.model_catalog,
            self.preview.clone(),
            self.crosshair.clone(),
            &self.parser_library,
        )
        .map(|_| ())
        .map_err(|error| PerceptionError::new(error.to_string()))
    }

    fn preflight_model(
        &self,
        candidate: &ModelCandidate,
    ) -> Result<Option<PerceptionModelContract>, PerceptionError> {
        let config = self.config.blocking_effective_snapshot();
        let model = self
            .model_catalog
            .runtime_artifact(candidate.project_id, candidate.artifact_id)
            .map_err(|error| PerceptionError::new(error.to_string()))?;
        resolve_model_nvinfer_config(
            &config,
            &model,
            Some(candidate.parser_preset.as_str()),
            &self.parser_library,
        )
        .map(|(_, contract)| Some(contract))
        .map_err(|error| PerceptionError::new(error.to_string()))
    }

    fn runtime_contract(&self) -> Result<Option<PerceptionRuntimeContract>, PerceptionError> {
        let config = self.config.blocking_effective_snapshot();
        let capture = config
            .require_vision_adapters()
            .map_err(|error| PerceptionError::new(error.to_string()))
            .and_then(|adapters| {
                resolve_capture_plan(adapters.capture)
                    .map_err(|error| PerceptionError::new(error.to_string()))
            })?;
        let model = active_runtime_model(&self.model_catalog)
            .map_err(|error| PerceptionError::new(error.to_string()))?;
        let model = resolve_model_nvinfer_config(&config, &model, None, &self.parser_library)
            .map(|(_, contract)| contract)
            .map_err(|error| PerceptionError::new(error.to_string()))?;
        Ok(Some(PerceptionRuntimeContract {
            model,
            source_width: capture.width,
            roi_width: capture.roi_width,
            roi_height: capture.roi_height,
        }))
    }

    fn start(
        &self,
        epoch: RuntimeEpoch,
        ingress: PipelineIngress,
        clock: Arc<dyn Clock>,
        events: std::sync::mpsc::SyncSender<PerceptionEvent>,
    ) -> Result<Box<dyn PerceptionSession>, PerceptionError> {
        let current = self.config.blocking_effective_snapshot();
        self.preview.configure(current.consumers.preview);
        let config = build_deepstream_session_config(
            &current,
            &self.model_catalog,
            self.preview.clone(),
            self.crosshair.clone(),
            &self.parser_library,
        )
        .map_err(|error| PerceptionError::new(error.to_string()))?;
        DeepStreamAdapter::with_latest_frames(config, self.latest_frames.clone())
            .start(epoch, ingress, clock, events)
    }
}

fn build_deepstream_session_config(
    config: &AppConfig,
    model_catalog: &SqliteModelCatalog,
    preview_hub: PreviewHub,
    crosshair_hub: Option<CrosshairHub>,
    parser_library: &Path,
) -> Result<DeepStreamSessionConfig, LivePerceptionError> {
    let adapters = config
        .require_vision_adapters()
        .map_err(LivePerceptionError::Config)?;
    let capture = resolve_capture_plan(adapters.capture)?;
    let format = parse_capture_format(&capture.pixel_format)?;
    let io_mode = u32::try_from(adapters.inference.deepstream_io_mode)
        .map_err(|_| LivePerceptionError::InvalidIoMode(adapters.inference.deepstream_io_mode))?;
    let (path, model_contract) =
        resolve_active_nvinfer_config(config, model_catalog, parser_library)?;
    let inference = if model_contract.preserves_roi_coordinates {
        InferenceStage::DeepStreamNvinferAspectPreserving { config: path }
    } else {
        InferenceStage::DeepStreamNvinfer { config: path }
    };
    let pipeline = DeepStreamPipelineSpec {
        device: capture.device.clone(),
        capture: CaptureProfile {
            width: capture.width,
            height: capture.height,
            fps: capture.fps,
            format,
        },
        io_mode,
        roi: Roi {
            left: capture.roi_left,
            top: capture.roi_top,
            width: capture.roi_width,
            height: capture.roi_height,
        },
        model_input: ModelInput {
            width: model_contract.input_width,
            height: model_contract.input_height,
        },
        inference,
        batched_push_timeout_us: adapters.inference.deepstream_batched_push_timeout_us,
        inference_element: adapters.inference.deepstream_probe_element.clone(),
        preview: adapters.consumers.preview.then_some(PreviewPipelineConfig {
            fps: adapters.limits.stream_fps,
        }),
        crosshair: config.crosshair.enabled.then_some(CrosshairPipelineConfig {
            size: config.crosshair.search_size,
            fps: config.crosshair.sample_hz,
        }),
    };
    pipeline
        .build()
        .map_err(|error| LivePerceptionError::Pipeline(error.to_string()))?;
    Ok(DeepStreamSessionConfig {
        pipeline,
        probe_pad: adapters.inference.deepstream_probe_pad.clone(),
        source_id: adapters.inference.deepstream_source_id,
        inference_component_id: adapters.inference.deepstream_component_id,
        max_batch_age_ns: Some(deadline_ns(adapters.inference.inference_input_deadline_ms)),
        startup_timeout: Duration::from_millis(adapters.inference.deepstream_startup_timeout_ms),
        shutdown_timeout: Duration::from_millis(adapters.inference.deepstream_shutdown_timeout_ms),
        preview: adapters.consumers.preview.then_some(preview_hub),
        crosshair: crosshair_hub,
    })
}

fn build_crosshair_hub(config: &AppConfig) -> Result<Option<CrosshairHub>, LivePerceptionError> {
    if !config.crosshair.enabled {
        return Ok(None);
    }
    let template_path = resolve_data_artifact(
        &config.paths.data_dir,
        Path::new("runtime/crosshair/template.json"),
    )?;
    Ok(Some(CrosshairHub::new(
        PipelineCrosshairConfig {
            enabled: config.crosshair.enabled,
            use_for_control: config.crosshair.use_for_control,
            search_size: config.crosshair.search_size,
            sample_frames: config.crosshair.sample_frames,
            confirm_duration: Duration::from_secs_f64(
                config.crosshair.confirm_duration_ms / 1_000.0,
            ),
            max_age: Duration::from_secs_f64(config.crosshair.max_age_ms / 1_000.0),
            max_offset_px: config.crosshair.max_offset_px,
            min_similarity: config.crosshair.min_similarity,
            max_step_px: config.crosshair.max_step_px,
        },
        template_path,
    )))
}

fn resolve_data_artifact(data_dir: &Path, artifact: &Path) -> Result<PathBuf, LivePerceptionError> {
    if artifact.is_absolute() {
        Ok(artifact.to_owned())
    } else {
        let data_dir = if data_dir.is_absolute() {
            data_dir.to_owned()
        } else {
            std::env::current_dir()
                .map_err(LivePerceptionError::CurrentDirectory)?
                .join(data_dir)
        };
        Ok(data_dir.join(artifact))
    }
}

fn resolve_process_path(path: &Path) -> Result<PathBuf, LivePerceptionError> {
    let resolved = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map_err(LivePerceptionError::CurrentDirectory)?
            .join(path)
    };
    if !resolved.is_file() {
        return Err(LivePerceptionError::ParserLibraryMissing(resolved));
    }
    Ok(resolved)
}

fn resolve_active_nvinfer_config(
    config: &AppConfig,
    model_catalog: &SqliteModelCatalog,
    parser_library: &Path,
) -> Result<(PathBuf, PerceptionModelContract), LivePerceptionError> {
    let model = active_runtime_model(model_catalog)?;
    resolve_model_nvinfer_config(config, &model, None, parser_library)
}

fn active_runtime_model(
    model_catalog: &SqliteModelCatalog,
) -> Result<RuntimeModelArtifact, LivePerceptionError> {
    let Some(active) = model_catalog
        .active_model()
        .map_err(LivePerceptionError::ModelCatalog)?
    else {
        return Err(LivePerceptionError::ActiveModelMissing);
    };
    Ok(RuntimeModelArtifact {
        project: active.project,
        version: active.version,
        artifact: active.artifact,
        artifact_path: active.artifact_path,
    })
}

fn resolve_model_nvinfer_config(
    config: &AppConfig,
    model: &RuntimeModelArtifact,
    requested_preset: Option<&str>,
    parser_library: &Path,
) -> Result<(PathBuf, PerceptionModelContract), LivePerceptionError> {
    let candidate_preflight = requested_preset.is_some();
    let inference = config
        .require_inference_adapter()
        .map_err(LivePerceptionError::Config)?;
    let parsed_manifest = validate_runtime_model(config, model)?;
    let manifest = &parsed_manifest.document;
    if manifest.postprocess.max_detections == 0 {
        return Err(manifest_error(
            "postprocess.max_detections must be positive",
        ));
    }
    let requested_preset = validate_parser_preset(
        requested_preset.unwrap_or(manifest.postprocess.parser_preset.as_str()),
        manifest.output.has_objectness,
    )
    .map_err(|error| manifest_error(error.message()))?;
    let parser_library = resolve_process_path(parser_library)?;
    let source = generate_nvinfer_config(
        manifest,
        &model.artifact_path,
        &parser_library,
        inference.deepstream_component_id,
        inference.confidence_threshold,
        inference.nms_threshold,
    )?;
    validate_nvinfer_manifest_contract(
        &source,
        manifest,
        inference.confidence_threshold,
        inference.nms_threshold,
    )?;
    let runtime_directory =
        resolve_data_artifact(&config.paths.data_dir, Path::new("runtime/deepstream"))?;
    fs::create_dir_all(&runtime_directory).map_err(|source| LivePerceptionError::WriteNvinfer {
        path: runtime_directory.clone(),
        source,
    })?;
    let path = if candidate_preflight {
        runtime_directory.join(format!("candidate-{}.nvinfer.ini", model.artifact.id))
    } else {
        runtime_directory.join("active-nvinfer.ini")
    };
    write_if_changed(&path, source.as_bytes())?;
    let parser = resolve_parser_contract(manifest)?;
    let input_shape = manifest
        .input
        .shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("x");
    let contract = PerceptionModelContract {
        input_shape,
        input_width: manifest_input_dimension(manifest, 3, "width")?,
        input_height: manifest_input_dimension(manifest, 2, "height")?,
        preserves_roi_coordinates: manifest.input.maintain_aspect_ratio,
        classes: manifest.output.class_names.clone(),
        parser: PerceptionParserContract {
            requested_preset,
            compatibility: if manifest.output.has_objectness {
                "yolov5".to_owned()
            } else {
                "yolov8_yolo11".to_owned()
            },
            has_objectness: manifest.output.has_objectness,
            parser_library: "novasight_builtin".to_owned(),
            parser_function: parser.function.to_owned(),
            nms_owner: "deepstream".to_owned(),
        },
    };
    Ok((path, contract))
}

fn manifest_input_dimension(
    manifest: &ModelManifest,
    index: usize,
    label: &'static str,
) -> Result<u32, LivePerceptionError> {
    manifest
        .input
        .shape
        .get(index)
        .copied()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| manifest_error(format!("input {label} exceeds runtime limits")))
}

fn validate_runtime_model(
    config: &AppConfig,
    model: &RuntimeModelArtifact,
) -> Result<ParsedModelManifest, LivePerceptionError> {
    config
        .require_inference_adapter()
        .map_err(LivePerceptionError::Config)?;
    if model.artifact.kind != "engine" {
        return Err(LivePerceptionError::ActiveArtifactKind {
            artifact_id: model.artifact.id,
            kind: model.artifact.kind.clone(),
        });
    }
    if model.artifact.status != "ready" {
        return Err(LivePerceptionError::ActiveArtifactNotReady {
            artifact_id: model.artifact.id,
            status: model.artifact.status.clone(),
        });
    }
    if !model.artifact_path.is_file() {
        return Err(LivePerceptionError::EngineMissing(
            model.artifact_path.clone(),
        ));
    }
    let parsed_manifest = read_model_manifest(&model.artifact_path)?;
    let manifest = &parsed_manifest.document;
    let actual_sha = validate_model_document(
        &model.artifact_path,
        manifest,
        parsed_manifest.output_class_names_present,
    )?;
    let registry_sha = normalize_registry_checksum(&model.artifact.checksum).ok_or_else(|| {
        LivePerceptionError::RegistryChecksumInvalid {
            artifact_id: model.artifact.id,
            checksum: model.artifact.checksum.clone(),
        }
    })?;
    if registry_sha != manifest.artifact.sha256.to_ascii_lowercase() || registry_sha != actual_sha {
        return Err(LivePerceptionError::RegistryChecksumMismatch {
            artifact_id: model.artifact.id,
            registry: registry_sha,
            manifest: manifest.artifact.sha256.clone(),
            actual: actual_sha,
        });
    }
    Ok(parsed_manifest)
}

fn normalize_registry_checksum(value: &str) -> Option<String> {
    let checksum = value.trim();
    let checksum = checksum
        .strip_prefix("sha256:")
        .or_else(|| checksum.strip_prefix("SHA256:"))
        .unwrap_or(checksum);
    (checksum.len() == 64 && checksum.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| checksum.to_ascii_lowercase())
}

fn generate_nvinfer_config(
    manifest: &ModelManifest,
    engine: &Path,
    parser_library: &Path,
    component_id: i32,
    confidence_threshold: f64,
    nms_threshold: f64,
) -> Result<String, LivePerceptionError> {
    let parser = resolve_parser_contract(manifest)?;
    let network_mode = match manifest
        .runtime
        .precision
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "fp32" => 0,
        "int8" => 1,
        "fp16" => 2,
        value => return Err(manifest_error(format!("unsupported precision {value}"))),
    };
    let color_format = match manifest
        .input
        .color_format
        .trim()
        .to_ascii_uppercase()
        .as_str()
    {
        "RGB" => 0,
        "BGR" => 1,
        "GRAY" | "GREY" => 2,
        value => return Err(manifest_error(format!("unsupported color format {value}"))),
    };
    let output_names = if manifest.output.bindings.is_empty() {
        manifest.output.name.clone()
    } else {
        manifest
            .output
            .bindings
            .iter()
            .map(|binding| binding.name.as_str())
            .collect::<Vec<_>>()
            .join(";")
    };
    let engine = engine
        .canonicalize()
        .map_err(|source| LivePerceptionError::ReadEngine {
            path: engine.to_owned(),
            source,
        })?;
    let parser_library = parser_library.canonicalize().map_err(|source| {
        LivePerceptionError::CanonicalizeParser {
            path: parser_library.to_owned(),
            source,
        }
    })?;
    let engine_value = safe_nvinfer_path("model-engine-file", &engine)?;
    let parser_value = safe_nvinfer_path("custom-lib-path", &parser_library)?;
    let source = format!(
        "# Generated by NovaSight Rust. Do not hand-edit.\n# novasight-model-fingerprint={}\n[property]\ngpu-id=0\nmodel-engine-file={}\nbatch-size=1\nnetwork-mode={}\nnetwork-type=0\nprocess-mode=1\ngie-unique-id={}\ninterval=0\nnum-detected-classes={}\nnet-scale-factor={:.17}\nmodel-color-format={}\nmaintain-aspect-ratio={}\nsymmetric-padding={}\noutput-tensor-meta=0\noutput-blob-names={}\ncustom-lib-path={}\nparse-bbox-func-name={}\ncluster-mode={}\n\n[class-attrs-all]\npre-cluster-threshold={:.8}\nnms-iou-threshold={:.8}\ntopk={}\n",
        manifest.model_fingerprint,
        engine_value,
        network_mode,
        component_id,
        manifest.output.class_count,
        manifest.input.scale_factor,
        color_format,
        i32::from(manifest.input.maintain_aspect_ratio),
        i32::from(manifest.input.symmetric_padding),
        output_names,
        parser_value,
        parser.function,
        parser.cluster_mode,
        confidence_threshold,
        nms_threshold,
        effective_deepstream_max_detections(manifest),
    );
    let values = parse_ini_values(&source);
    validate_ini_string(&values, "model-engine-file", engine_value)?;
    validate_ini_string(&values, "custom-lib-path", parser_value)?;
    Ok(source)
}

fn safe_nvinfer_path<'a>(
    label: &'static str,
    path: &'a Path,
) -> Result<&'a str, LivePerceptionError> {
    let value = path
        .to_str()
        .filter(|value| !value.contains(['\r', '\n']))
        .ok_or_else(|| LivePerceptionError::InvalidNvinferPath {
            label,
            path: path.to_owned(),
        })?;
    Ok(value)
}

fn write_if_changed(path: &Path, contents: &[u8]) -> Result<(), LivePerceptionError> {
    if fs::read(path).ok().as_deref() == Some(contents) {
        return Ok(());
    }
    let temporary = path.with_extension(format!("ini.{}.tmp", std::process::id()));
    fs::write(&temporary, contents).map_err(|source| LivePerceptionError::WriteNvinfer {
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, path).map_err(|source| LivePerceptionError::WriteNvinfer {
        path: path.to_owned(),
        source,
    })
}

fn parse_capture_format(value: &str) -> Result<CaptureFormat, LivePerceptionError> {
    match value.trim().to_ascii_uppercase().as_str() {
        "MJPEG" | "MJPG" => Ok(CaptureFormat::Mjpeg),
        "NV12" => Ok(CaptureFormat::Nv12),
        "YUY2" | "YUYV" => Ok(CaptureFormat::Yuy2),
        _ => Err(LivePerceptionError::UnsupportedCaptureFormat(
            value.to_owned(),
        )),
    }
}

fn deadline_ns(milliseconds: f64) -> u64 {
    (milliseconds * 1_000_000.0)
        .round()
        .clamp(1.0, u64::MAX as f64) as u64
}

fn parse_ini_values(source: &str) -> BTreeMap<String, String> {
    source
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty()
                || line.starts_with('#')
                || line.starts_with(';')
                || line.starts_with('[')
            {
                return None;
            }
            let (key, value) = line.split_once('=')?;
            Some((key.trim().to_owned(), value.trim().to_owned()))
        })
        .collect()
}

struct ParsedModelManifest {
    document: ModelManifest,
    output_class_names_present: bool,
}

fn read_model_manifest(engine: &Path) -> Result<ParsedModelManifest, LivePerceptionError> {
    let file_name = engine
        .file_name()
        .ok_or_else(|| LivePerceptionError::EngineFileName(engine.to_owned()))?;
    let manifest = engine.with_file_name(format!("{}.manifest.json", file_name.to_string_lossy()));
    let source =
        fs::read_to_string(&manifest).map_err(|source| LivePerceptionError::ReadManifest {
            path: manifest.clone(),
            source,
        })?;
    let raw: serde_json::Value =
        serde_json::from_str(&source).map_err(|source| LivePerceptionError::ParseManifest {
            path: manifest.clone(),
            source,
        })?;
    let output_class_names_present = raw
        .get("output")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|output| output.contains_key("class_names"));
    let document =
        serde_json::from_value(raw).map_err(|source| LivePerceptionError::ParseManifest {
            path: manifest,
            source,
        })?;
    Ok(ParsedModelManifest {
        document,
        output_class_names_present,
    })
}

fn validate_model_document(
    engine: &Path,
    document: &ModelManifest,
    output_class_names_present: bool,
) -> Result<String, LivePerceptionError> {
    validate_manifest_shape(document)?;
    if !document.validated {
        return Err(manifest_error("model manifest must have validated=true"));
    }
    let fingerprint = require_nonempty(&document.model_fingerprint, "model_fingerprint")?;
    let computed_fingerprint = compute_model_fingerprint(document)?;
    let legacy_fingerprint = (!output_class_names_present)
        .then(|| compute_model_fingerprint_with_class_names(document, false))
        .transpose()?;
    if fingerprint != computed_fingerprint && legacy_fingerprint.as_deref() != Some(fingerprint) {
        return Err(manifest_error(format!(
            "model_fingerprint {fingerprint} does not match canonical content {computed_fingerprint}"
        )));
    }
    validate_manifest_probability(
        "postprocess.confidence_threshold",
        document.postprocess.confidence_threshold,
    )?;
    validate_manifest_probability(
        "postprocess.nms_iou_threshold",
        document.postprocess.nms_iou_threshold,
    )?;
    let engine_name = engine
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| manifest_error("engine file name is not valid UTF-8"))?;
    if document.artifact.engine_path != engine_name {
        return Err(manifest_error(format!(
            "artifact.engine_path {} does not match engine file {engine_name}",
            document.artifact.engine_path
        )));
    }
    let actual_size = fs::metadata(engine)
        .map_err(|source| LivePerceptionError::ReadEngine {
            path: engine.to_owned(),
            source,
        })?
        .len();
    if actual_size != document.artifact.size_bytes {
        return Err(manifest_error(format!(
            "engine size {actual_size} does not match manifest {}",
            document.artifact.size_bytes
        )));
    }
    let actual_sha = sha256_file(engine)?;
    if actual_sha != document.artifact.sha256 {
        return Err(manifest_error(format!(
            "engine sha256 {actual_sha} does not match manifest {}",
            document.artifact.sha256
        )));
    }
    Ok(actual_sha)
}

fn validate_manifest_probability(
    field: &'static str,
    actual: f64,
) -> Result<(), LivePerceptionError> {
    if !actual.is_finite() || !(0.0..=1.0).contains(&actual) {
        return Err(manifest_error(format!(
            "manifest {field} must be in [0, 1]"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ParserContract {
    function: &'static str,
    cluster_mode: i64,
}

fn validate_manifest_shape(manifest: &ModelManifest) -> Result<(), LivePerceptionError> {
    if manifest.schema_version != 1 {
        return Err(manifest_error(format!(
            "unsupported schema_version {}; expected 1",
            manifest.schema_version
        )));
    }
    require_nonempty(&manifest.model_id, "model_id")?;
    require_nonempty(&manifest.display_name, "display_name")?;
    require_nonempty(&manifest.artifact.engine_path, "artifact.engine_path")?;
    require_nonempty(&manifest.artifact.sha256, "artifact.sha256")?;
    for (label, tensor) in [
        (
            "input",
            ManifestTensorRef {
                name: &manifest.input.name,
                shape: &manifest.input.shape,
                dtype: &manifest.input.dtype,
                layout: &manifest.input.layout,
            },
        ),
        (
            "output",
            ManifestTensorRef {
                name: &manifest.output.name,
                shape: &manifest.output.shape,
                dtype: &manifest.output.dtype,
                layout: &manifest.output.layout,
            },
        ),
    ] {
        validate_tensor(label, tensor)?;
    }
    for binding in &manifest.output.bindings {
        validate_tensor(
            "output.bindings",
            ManifestTensorRef {
                name: &binding.name,
                shape: &binding.shape,
                dtype: &binding.dtype,
                layout: &binding.layout,
            },
        )?;
    }
    if manifest.runtime.batch_size != 1 {
        return Err(manifest_error("runtime.batch_size must be 1"));
    }
    if !manifest.input.layout.eq_ignore_ascii_case("NCHW")
        || manifest.input.shape.len() != 4
        || manifest.input.shape[0] != 1
    {
        return Err(manifest_error(format!(
            "input {:?} {} must be batch-one NCHW with positive dimensions",
            manifest.input.shape, manifest.input.layout
        )));
    }
    if manifest.output.class_count == 0 {
        return Err(manifest_error("output.class_count must be positive"));
    }
    Ok(())
}

struct ManifestTensorRef<'a> {
    name: &'a str,
    shape: &'a [u64],
    dtype: &'a str,
    layout: &'a str,
}

fn validate_tensor(label: &str, tensor: ManifestTensorRef<'_>) -> Result<(), LivePerceptionError> {
    require_nonempty(tensor.name, &format!("{label}.name"))?;
    require_nonempty(tensor.dtype, &format!("{label}.dtype"))?;
    require_nonempty(tensor.layout, &format!("{label}.layout"))?;
    if tensor.shape.is_empty() || tensor.shape.contains(&0) {
        return Err(manifest_error(format!(
            "{label}.shape must contain positive dimensions"
        )));
    }
    Ok(())
}

fn require_nonempty<'a>(value: &'a str, field: &str) -> Result<&'a str, LivePerceptionError> {
    let value = value.trim();
    if value.is_empty() {
        Err(manifest_error(format!("{field} must not be empty")))
    } else {
        Ok(value)
    }
}

fn compute_model_fingerprint(manifest: &ModelManifest) -> Result<String, LivePerceptionError> {
    compute_model_fingerprint_with_class_names(manifest, true)
}

fn compute_model_fingerprint_with_class_names(
    manifest: &ModelManifest,
    include_class_names: bool,
) -> Result<String, LivePerceptionError> {
    let mut output = serde_json::json!({
        "name": manifest.output.name,
        "shape": manifest.output.shape,
        "dtype": manifest.output.dtype,
        "layout": manifest.output.layout,
        "format": manifest.output.format,
        "class_count": manifest.output.class_count,
        "has_objectness": manifest.output.has_objectness,
        "coordinate_mode": manifest.output.coordinate_mode,
    });
    let output = output
        .as_object_mut()
        .expect("json object construction is infallible");
    if include_class_names {
        output.insert(
            "class_names".to_owned(),
            serde_json::json!(manifest.output.class_names),
        );
    }
    if !manifest.output.bindings.is_empty() {
        output.insert(
            "bindings".to_owned(),
            serde_json::to_value(&manifest.output.bindings)
                .map_err(LivePerceptionError::SerializeFingerprint)?,
        );
    }
    if !manifest.output.strides.is_empty() {
        output.insert(
            "strides".to_owned(),
            serde_json::json!(manifest.output.strides),
        );
    }
    if !manifest.output.anchors.is_empty() {
        output.insert(
            "anchors".to_owned(),
            serde_json::json!(manifest.output.anchors),
        );
    }
    let payload = serde_json::json!({
        "artifact_sha256": manifest.artifact.sha256,
        "input": {
            "name": manifest.input.name,
            "shape": manifest.input.shape,
            "dtype": manifest.input.dtype,
            "layout": manifest.input.layout,
        },
        "output": output,
        "parser_schema": "yolo-v1",
    });
    let stable =
        serde_json::to_string(&payload).map_err(LivePerceptionError::SerializeFingerprint)?;
    let stable = python_json_numbers(&stable);
    Ok(sha256_bytes(stable.as_bytes()))
}

/// Match Python's stable_json float spelling at the two ryu differences used
/// by manifests: exponent padding and scientific notation below `1e-4`.
/// Only JSON number tokens outside strings are rewritten.
fn python_json_numbers(json: &str) -> String {
    let bytes = json.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            output.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
            output.push(byte);
            index += 1;
            continue;
        }
        if !byte.is_ascii_digit() && byte != b'-' {
            output.push(byte);
            index += 1;
            continue;
        }
        let number_start = index;
        index += 1;
        while index < bytes.len()
            && (bytes[index].is_ascii_digit()
                || matches!(bytes[index], b'.' | b'e' | b'E' | b'+' | b'-'))
        {
            index += 1;
        }
        output.extend_from_slice(python_number_token(&bytes[number_start..index]).as_bytes());
    }
    String::from_utf8(output).expect("valid JSON remains UTF-8")
}

fn python_number_token(token: &[u8]) -> String {
    let token = std::str::from_utf8(token).expect("JSON numbers are ASCII");
    if let Some(exponent_at) = token.find(['e', 'E']) {
        let (mantissa, exponent) = token.split_at(exponent_at);
        let exponent = &exponent[1..];
        let (sign, digits) = exponent
            .strip_prefix(['+', '-'])
            .map_or(("", exponent), |digits| (&exponent[..1], digits));
        return format!(
            "{mantissa}e{sign}{}{digits}",
            if digits.len() == 1 { "0" } else { "" }
        );
    }
    let (sign, unsigned) = token
        .strip_prefix('-')
        .map_or(("", token), |value| ("-", value));
    let Some(fraction) = unsigned.strip_prefix("0.") else {
        return token.to_owned();
    };
    let leading_zeros = fraction.bytes().take_while(|byte| *byte == b'0').count();
    if leading_zeros < 4 || leading_zeros == fraction.len() {
        return token.to_owned();
    }
    let significant = &fraction[leading_zeros..];
    let (first, rest) = significant.split_at(1);
    let mantissa = if rest.is_empty() {
        first.to_owned()
    } else {
        format!("{first}.{rest}")
    };
    let exponent = leading_zeros + 1;
    format!(
        "{sign}{mantissa}e-{}{exponent}",
        if exponent < 10 { "0" } else { "" }
    )
}

fn validate_nvinfer_manifest_contract(
    source: &str,
    manifest: &ModelManifest,
    confidence_threshold: f64,
    nms_threshold: f64,
) -> Result<(), LivePerceptionError> {
    let values = parse_ini_values(source);
    let parser = resolve_parser_contract(manifest)?;
    validate_ini_integer(&values, "batch-size", 1)?;
    validate_ini_integer(&values, "network-type", 0)?;
    validate_ini_integer(&values, "process-mode", 1)?;
    validate_ini_integer(&values, "interval", 0)?;
    validate_ini_integer(
        &values,
        "num-detected-classes",
        i64::from(manifest.output.class_count),
    )?;
    validate_ini_integer(&values, "output-tensor-meta", 0)?;
    validate_ini_integer(&values, "cluster-mode", parser.cluster_mode)?;
    validate_ini_integer(
        &values,
        "network-mode",
        match manifest
            .runtime
            .precision
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "fp32" => 0,
            "int8" => 1,
            "fp16" => 2,
            value => return Err(manifest_error(format!("unsupported precision {value}"))),
        },
    )?;
    validate_ini_integer(
        &values,
        "model-color-format",
        match manifest
            .input
            .color_format
            .trim()
            .to_ascii_uppercase()
            .as_str()
        {
            "RGB" => 0,
            "BGR" => 1,
            "GRAY" | "GREY" => 2,
            value => return Err(manifest_error(format!("unsupported color format {value}"))),
        },
    )?;
    validate_ini_integer(
        &values,
        "maintain-aspect-ratio",
        i64::from(manifest.input.maintain_aspect_ratio),
    )?;
    validate_ini_integer(
        &values,
        "symmetric-padding",
        i64::from(manifest.input.symmetric_padding),
    )?;
    validate_ini_integer(
        &values,
        "topk",
        i64::from(effective_deepstream_max_detections(manifest)),
    )?;
    validate_ini_float(&values, "net-scale-factor", manifest.input.scale_factor)?;
    validate_ini_float(&values, "pre-cluster-threshold", confidence_threshold)?;
    validate_ini_float(&values, "nms-iou-threshold", nms_threshold)?;
    validate_ini_string(&values, "parse-bbox-func-name", parser.function)?;
    let output_names = if manifest.output.bindings.is_empty() {
        manifest.output.name.clone()
    } else {
        manifest
            .output
            .bindings
            .iter()
            .map(|binding| binding.name.as_str())
            .collect::<Vec<_>>()
            .join(";")
    };
    validate_ini_string(&values, "output-blob-names", &output_names)?;
    Ok(())
}

fn effective_deepstream_max_detections(manifest: &ModelManifest) -> u32 {
    manifest
        .postprocess
        .max_detections
        .min(DEEPSTREAM_MAX_DETECTIONS as u32)
}

fn resolve_parser_contract(
    manifest: &ModelManifest,
) -> Result<ParserContract, LivePerceptionError> {
    let parser = manifest.postprocess.parser.trim().to_ascii_lowercase();
    let format = manifest.output.format.trim().to_ascii_lowercase();
    let preset = normalize_parser_preset(&manifest.postprocess.parser_preset)?;
    match parser.as_str() {
        "decoded_nms" => {
            if format != "decoded_boxes6"
                || manifest.output.shape.last().copied() != Some(6)
                || manifest.output.shape.iter().product::<u64>() % 6 != 0
            {
                return Err(manifest_error(
                    "decoded_nms requires a positive output shape whose last dimension is 6",
                ));
            }
            Ok(ParserContract {
                function: "NvDsInferParseNovaSightDecodedNms",
                cluster_mode: 4,
            })
        }
        "rockchip_yolov5" => {
            if format != "rockchip_yolov5_three_scale"
                || manifest.output.bindings.len() != 3
                || manifest.output.strides != [8, 16, 32]
                || manifest.output.anchors.len() != 3
                || manifest.output.anchors.iter().any(|scale| scale.len() != 6)
                || matches!(preset, "yolov8" | "yolo11")
            {
                return Err(manifest_error("invalid Rockchip YOLOv5 parser contract"));
            }
            Ok(ParserContract {
                function: "NvDsInferParseNovaSightRockchipYoloV5",
                cluster_mode: 2,
            })
        }
        "efficientnms" => {
            if format != "efficientnms_boxes_scores_classes" || manifest.output.bindings.len() != 4
            {
                return Err(manifest_error("invalid EfficientNMS parser contract"));
            }
            Ok(ParserContract {
                function: "NvDsInferParseNovaSightEfficientNms",
                cluster_mode: 4,
            })
        }
        "yolo" => {
            if format != "yolo_cxcywh_class_scores"
                || !manifest
                    .output
                    .coordinate_mode
                    .eq_ignore_ascii_case("pixel")
            {
                return Err(manifest_error("invalid raw YOLO output contract"));
            }
            let shape = match manifest.output.shape.as_slice() {
                [1, left, right] => [*left, *right],
                [left, right] => [*left, *right],
                _ => {
                    return Err(manifest_error(
                        "YOLO output must be [C,N], [N,C], or [1,C,N]",
                    ));
                }
            };
            let preset_objectness = match preset {
                "yolov5" => Some(true),
                "yolov8" | "yolo11" => Some(false),
                _ => None,
            };
            if preset_objectness.is_some_and(|value| value != manifest.output.has_objectness) {
                return Err(manifest_error(
                    "parser preset conflicts with output objectness",
                ));
            }
            let has_objectness = preset_objectness.unwrap_or(manifest.output.has_objectness);
            let channels =
                u64::from(manifest.output.class_count) + if has_objectness { 5 } else { 4 };
            if !shape.contains(&channels) || shape.iter().max().copied().unwrap_or(0) <= channels {
                return Err(manifest_error(
                    "YOLO output shape does not match class/objectness contract",
                ));
            }
            Ok(ParserContract {
                function: "NvDsInferParseNovaSightRaw",
                cluster_mode: 2,
            })
        }
        _ => Err(manifest_error(format!("unsupported parser {parser}"))),
    }
}

fn normalize_parser_preset(value: &str) -> Result<&str, LivePerceptionError> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    let canonical = match normalized.as_str() {
        "" | "automatic" => "auto",
        "yolo_v5" | "yolov5_raw" => "yolov5",
        "yolo_v8" | "yolov8_raw" => "yolov8",
        "yolo_11" | "yolo11_raw" => "yolo11",
        "generic" | "custom" | "novasight" => "novasight_generic",
        "auto" | "yolov5" | "yolov8" | "yolo11" | "novasight_generic" => normalized.as_str(),
        _ => return Err(manifest_error(format!("unsupported parser preset {value}"))),
    };
    Ok(match canonical {
        "auto" => "auto",
        "yolov5" => "yolov5",
        "yolov8" => "yolov8",
        "yolo11" => "yolo11",
        _ => "novasight_generic",
    })
}

fn validate_ini_string(
    values: &BTreeMap<String, String>,
    key: &'static str,
    expected: &str,
) -> Result<(), LivePerceptionError> {
    let actual = values
        .get(key)
        .ok_or_else(|| manifest_error(format!("nvinfer is missing {key}")))?;
    if actual != expected {
        return Err(manifest_error(format!(
            "nvinfer {key}={actual} does not match manifest {expected}"
        )));
    }
    Ok(())
}

fn validate_ini_integer(
    values: &BTreeMap<String, String>,
    key: &'static str,
    expected: i64,
) -> Result<(), LivePerceptionError> {
    let actual = values
        .get(key)
        .and_then(|value| value.parse::<i64>().ok())
        .ok_or_else(|| manifest_error(format!("nvinfer has no valid {key}")))?;
    if actual != expected {
        return Err(manifest_error(format!(
            "nvinfer {key}={actual} does not match manifest {expected}"
        )));
    }
    Ok(())
}

fn validate_ini_float(
    values: &BTreeMap<String, String>,
    key: &'static str,
    expected: f64,
) -> Result<(), LivePerceptionError> {
    let actual = values
        .get(key)
        .and_then(|value| value.parse::<f64>().ok())
        .ok_or_else(|| manifest_error(format!("nvinfer has no valid {key}")))?;
    if !actual.is_finite() || (actual - expected).abs() > 1e-12 {
        return Err(manifest_error(format!(
            "nvinfer {key}={actual} does not match manifest {expected}"
        )));
    }
    Ok(())
}

fn manifest_error(message: impl Into<String>) -> LivePerceptionError {
    LivePerceptionError::ManifestContract(message.into())
}

fn sha256_file(path: &Path) -> Result<String, LivePerceptionError> {
    let mut file = fs::File::open(path).map_err(|source| LivePerceptionError::ReadEngine {
        path: path.to_owned(),
        source,
    })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|source| LivePerceptionError::ReadEngine {
                path: path.to_owned(),
                source,
            })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(encode_sha256(digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    encode_sha256(digest.finalize())
}

fn encode_sha256(digest: impl IntoIterator<Item = u8>) -> String {
    let digest = digest.into_iter();
    let mut encoded = String::with_capacity(digest.size_hint().0 * 2);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use novasight_store::model_manifest::{
        ManifestArtifact, ManifestInput, ManifestOutput, ManifestPostprocess, ManifestRuntime,
    };

    fn canonical_manifest() -> ModelManifest {
        ModelManifest {
            schema_version: 1,
            model_id: "detector".to_owned(),
            display_name: "Detector".to_owned(),
            artifact: ManifestArtifact {
                engine_path: "detector.engine".to_owned(),
                sha256: "ab".repeat(32),
                size_bytes: 123,
            },
            runtime: ManifestRuntime {
                backend: "custom_tensorrt".to_owned(),
                precision: "fp16".to_owned(),
                batch_size: 1,
            },
            input: ManifestInput {
                name: "images".to_owned(),
                shape: vec![1, 3, 640, 640],
                dtype: "float32".to_owned(),
                layout: "NCHW".to_owned(),
                color_format: "RGB".to_owned(),
                scale_factor: 1.0 / 255.0,
                maintain_aspect_ratio: false,
                symmetric_padding: false,
            },
            output: ManifestOutput {
                name: "output0".to_owned(),
                shape: vec![1, 84, 8400],
                dtype: "float32".to_owned(),
                layout: "NCHW".to_owned(),
                format: "yolo_cxcywh_class_scores".to_owned(),
                class_count: 80,
                class_names: (0..80).map(|value| value.to_string()).collect(),
                has_objectness: false,
                scores_are_sigmoid: true,
                coordinate_mode: "pixel".to_owned(),
                bindings: Vec::new(),
                strides: Vec::new(),
                anchors: Vec::new(),
            },
            postprocess: ManifestPostprocess {
                parser: "yolo".to_owned(),
                parser_preset: "yolov8".to_owned(),
                confidence_threshold: 0.25,
                nms_iou_threshold: 0.45,
                class_aware_nms: true,
                max_detections: 300,
            },
            validated: true,
            model_fingerprint: String::new(),
        }
    }

    #[test]
    fn fingerprint_matches_python_canonical_json_contract() {
        assert_eq!(
            compute_model_fingerprint(&canonical_manifest()).expect("fingerprint"),
            "3866800678bc7e9031b148c4ec8f5314457c93f211f780557abaea5212e56d9b"
        );
    }

    #[test]
    fn fingerprint_matches_python_exponent_float_format() {
        let mut manifest = canonical_manifest();
        manifest.output.anchors = vec![vec![1e-7]];
        assert_eq!(
            compute_model_fingerprint(&manifest).expect("fingerprint"),
            "3baa62909243392b4bd0aeec3a23726a23112973a346654f5f6b9fd3b280858d"
        );
        assert_eq!(
            python_json_numbers(r#"{"value":1e-7,"text":"1e-7"}"#),
            r#"{"value":1e-07,"text":"1e-7"}"#
        );
    }

    #[test]
    fn fingerprint_matches_python_small_fixed_float_format() {
        let mut manifest = canonical_manifest();
        manifest.output.anchors = vec![vec![1e-5]];
        assert_eq!(
            compute_model_fingerprint(&manifest).expect("fingerprint"),
            "efdfa2dc09af968ea57a7a7bcbf9c098a7e83335897e8b2e95574dfbbdeb5696"
        );
        assert_eq!(python_number_token(b"0.00001234"), "1.234e-05");
    }

    #[test]
    fn registry_checksum_accepts_python_prefix_and_normalizes_hex_case() {
        let checksum = "AB".repeat(32);
        assert_eq!(
            normalize_registry_checksum(&format!("sha256:{checksum}")),
            Some("ab".repeat(32))
        );
        assert_eq!(normalize_registry_checksum("sha256:abc"), None);
        assert_eq!(normalize_registry_checksum(&"z".repeat(64)), None);
    }

    #[test]
    fn legacy_fingerprint_can_omit_defaulted_class_names() {
        assert_eq!(
            compute_model_fingerprint_with_class_names(&canonical_manifest(), false)
                .expect("legacy fingerprint"),
            "e10aff7d0417c57aed26c2bf1c8f405b5f22cecade74ec2df3515cbbe0c47ae6"
        );
    }

    #[test]
    fn nvinfer_paths_reject_line_injection() {
        assert!(safe_nvinfer_path("engine", Path::new("/tmp/model\ninterval=99")).is_err());
        assert_eq!(
            safe_nvinfer_path("engine", Path::new("/tmp/model.engine")).unwrap(),
            "/tmp/model.engine"
        );
    }

    #[test]
    fn nvinfer_contract_rejects_parser_semantic_drift() {
        let manifest = canonical_manifest();
        let valid = format!(
            "batch-size=1\nnetwork-mode=2\nnetwork-type=0\nprocess-mode=1\ninterval=0\nnum-detected-classes=80\nnet-scale-factor={:.17}\nmodel-color-format=0\nmaintain-aspect-ratio=0\nsymmetric-padding=0\noutput-tensor-meta=0\noutput-blob-names=output0\nparse-bbox-func-name=NvDsInferParseNovaSightRaw\ncluster-mode=2\npre-cluster-threshold=0.30\nnms-iou-threshold=0.50\ntopk=300\n",
            1.0 / 255.0
        );
        validate_nvinfer_manifest_contract(&valid, &manifest, 0.30, 0.50)
            .expect("canonical contract");
        let drifted = valid.replace(
            "NvDsInferParseNovaSightRaw",
            "NvDsInferParseNovaSightEfficientNms",
        );
        assert!(validate_nvinfer_manifest_contract(&drifted, &manifest, 0.30, 0.50).is_err());
    }
}

#[derive(Debug, Error)]
pub(super) enum LivePerceptionError {
    #[error("production adapter configuration is invalid: {0}")]
    Config(ConfigValidationError),
    #[error("model catalog failed: {0}")]
    ModelCatalog(novasight_store::model_catalog::ModelCatalogError),
    #[error("no active model deployment; publish a ready engine before starting perception")]
    ActiveModelMissing,
    #[error("active model artifact {artifact_id} must be an engine, got {kind}")]
    ActiveArtifactKind { artifact_id: i64, kind: String },
    #[error("active model artifact {artifact_id} is not ready: {status}")]
    ActiveArtifactNotReady { artifact_id: i64, status: String },
    #[error("active model artifact {artifact_id} has invalid registry checksum {checksum}")]
    RegistryChecksumInvalid { artifact_id: i64, checksum: String },
    #[error(
        "active model artifact {artifact_id} checksum mismatch: registry={registry}, manifest={manifest}, actual={actual}"
    )]
    RegistryChecksumMismatch {
        artifact_id: i64,
        registry: String,
        manifest: String,
        actual: String,
    },
    #[error("live DeepStream requires inference.enabled=true")]
    InferenceDisabled,
    #[error("capture capability probe failed: {0}")]
    CaptureProbe(String),
    #[error("capture profile cannot be resolved for the selected Jetson adapter: {0}")]
    CaptureSelection(String),
    #[error("hardware.host must be an IPv4 address for native kmNet: {0}")]
    InvalidKmNetHost(String),
    #[error("native kmNet production adapter failed: {0}")]
    NativeKmNet(KmNetNativeError),
    #[error("unsupported capture pixel format: {0}")]
    UnsupportedCaptureFormat(String),
    #[error("invalid DeepStream I/O mode: {0}")]
    InvalidIoMode(i32),
    #[error("DeepStream parser library does not exist: {}", .0.display())]
    ParserLibraryMissing(PathBuf),
    #[error("failed to read current working directory: {0}")]
    CurrentDirectory(std::io::Error),
    #[error("failed to write generated nvinfer configuration {}: {source}", path.display())]
    WriteNvinfer {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to resolve DeepStream parser library {}: {source}", path.display())]
    CanonicalizeParser {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("generated nvinfer {label} path is not safe UTF-8: {}", path.display())]
    InvalidNvinferPath { label: &'static str, path: PathBuf },
    #[error("TensorRT engine does not exist: {}", .0.display())]
    EngineMissing(PathBuf),
    #[error("TensorRT engine path has no file name: {}", .0.display())]
    EngineFileName(PathBuf),
    #[error("failed to read model manifest {}: {source}", path.display())]
    ReadManifest {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse model manifest {}: {source}", path.display())]
    ParseManifest {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to serialize canonical model fingerprint: {0}")]
    SerializeFingerprint(serde_json::Error),
    #[error("failed to read TensorRT engine {}: {source}", path.display())]
    ReadEngine {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("model deployment contract invalid: {0}")]
    ManifestContract(String),
    #[error("DeepStream pipeline configuration is invalid: {0}")]
    Pipeline(String),
    #[error("DeepStream native runtime preflight failed: {0}")]
    RuntimePreflight(SessionError),
}
