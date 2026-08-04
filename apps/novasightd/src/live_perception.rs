//! Jetson production adapter composition.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, UNIX_EPOCH};

use novasight_core::{
    CaptureCapabilityProbe, CaptureSelectionPreference, Clock, PointerDevice, RuntimeEpoch,
    select_capture_profile_for_formats,
};
use novasight_pipeline::{
    CrosshairConfig as PipelineCrosshairConfig, CrosshairHub, ModelCandidate,
    ParserContract as PerceptionParserContract, PerceptionAdapter, PerceptionError,
    PerceptionEvent, PerceptionModelContract, PerceptionRuntimeContract, PerceptionSession,
    PipelineIngress, PreviewHub, validate_parser_preset,
};
use novasight_platform_jetson::SystemMonotonicClock;
use novasight_platform_jetson::deepstream::{
    CaptureFormat, CaptureProfile, CrosshairPipelineConfig, DeepStreamAdapter,
    DeepStreamPipelineSpec, DeepStreamSessionConfig, InferenceStage, LatestFrameExchange,
    ModelInput, PreviewPipelineConfig, Roi, SessionError, deepstream_parser_contract,
    preflight_deepstream_runtime, render_deepstream_nvinfer_config,
};
use novasight_platform_jetson::kmnet::{KmNetNativeConfig, KmNetNativeDevice, KmNetNativeError};
use novasight_platform_jetson::v4l2::V4l2CapabilityProbe;
use novasight_runtime::{ConfigService, RuntimeDependencies, compose_pipeline_config};
use novasight_store::config::{
    AppConfig, CaptureConfig, CapturePreference, ConfigValidationError, DeviceBackend, DeviceConfig,
};
use novasight_store::model_catalog::{RuntimeModelArtifact, SqliteModelCatalog};
use novasight_store::model_manifest::ModelManifest;
#[cfg(test)]
use novasight_store::model_manifest::{compute_model_fingerprint, model_fingerprint_matches};
use novasight_store::model_runtime_contract::validate_model_runtime_contract;
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
    let session = build_deepstream_session_config(
        config,
        model_catalog,
        preview,
        crosshair,
        parser_library,
        &ModelIdentityCache::default(),
    )?;
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

#[derive(Clone, Debug, Default)]
struct ModelIdentityCache {
    entries: Arc<Mutex<HashMap<PathBuf, CachedModelIdentity>>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ModelFileStamp {
    size_bytes: u64,
    modified_ns: u128,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
}

#[derive(Clone, Debug)]
struct CachedModelIdentity {
    stamp: ModelFileStamp,
    sha256: String,
}

impl ModelIdentityCache {
    fn sha256(&self, path: &Path) -> Result<String, LivePerceptionError> {
        let canonical = path
            .canonicalize()
            .map_err(|source| LivePerceptionError::ReadEngine {
                path: path.to_owned(),
                source,
            })?;
        let before =
            fs::metadata(&canonical).map_err(|source| LivePerceptionError::ReadEngine {
                path: canonical.clone(),
                source,
            })?;
        let stamp = ModelFileStamp::from_metadata(&before);
        if let Some(cached) = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&canonical)
            .filter(|cached| cached.stamp == stamp)
        {
            return Ok(cached.sha256.clone());
        }

        let sha256 = sha256_file(&canonical)?;
        let after = fs::metadata(&canonical).map_err(|source| LivePerceptionError::ReadEngine {
            path: canonical.clone(),
            source,
        })?;
        if ModelFileStamp::from_metadata(&after) != stamp {
            return Err(manifest_error(
                "engine file changed while its identity was being validated",
            ));
        }
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                canonical,
                CachedModelIdentity {
                    stamp,
                    sha256: sha256.clone(),
                },
            );
        Ok(sha256)
    }
}

impl ModelFileStamp {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            size_bytes: metadata.len(),
            modified_ns: metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_nanos())
                .unwrap_or(0),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            changed_seconds: metadata.ctime(),
            #[cfg(unix)]
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

fn build_live_dependencies(
    config: &AppConfig,
    config_service: ConfigService,
    model_catalog: SqliteModelCatalog,
    device: Arc<dyn PointerDevice>,
    trigger_poll_interval_ms: Option<u64>,
    parser_library: PathBuf,
) -> Result<RuntimeDependencies, LivePerceptionError> {
    let clock: Arc<dyn Clock> = Arc::new(SystemMonotonicClock::default());
    let latest_frames = LatestFrameExchange::new();
    let preview = PreviewHub::new(config.consumers.preview);
    let crosshair = build_crosshair_hub(config)?;
    let pipeline = compose_pipeline_config(config, trigger_poll_interval_ms)
        .map_err(LivePerceptionError::Pipeline)?;
    let mut dependencies = RuntimeDependencies::new(clock, device, pipeline)
        .with_model_catalog(model_catalog.clone())
        .with_preview(preview.clone())
        .with_perception(Arc::new(CatalogDeepStreamAdapter {
            config: config_service,
            model_catalog,
            latest_frames,
            preview,
            crosshair: crosshair.clone(),
            parser_library,
            model_identity_cache: ModelIdentityCache::default(),
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
    model_identity_cache: ModelIdentityCache,
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
            &self.model_identity_cache,
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
            &self.model_identity_cache,
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
        let model = active_runtime_model(&self.model_catalog).map_err(perception_error)?;
        let model = resolve_model_nvinfer_config(
            &config,
            &model,
            None,
            &self.parser_library,
            &self.model_identity_cache,
        )
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
            &self.model_identity_cache,
        )
        .map_err(|error| PerceptionError::new(error.to_string()))?;
        DeepStreamAdapter::with_latest_frames(config, self.latest_frames.clone())
            .start(epoch, ingress, clock, events)
    }
}

fn perception_error(error: LivePerceptionError) -> PerceptionError {
    match error {
        LivePerceptionError::ActiveModelMissing => {
            PerceptionError::active_model_missing(error.to_string())
        }
        error => PerceptionError::new(error.to_string()),
    }
}

fn build_deepstream_session_config(
    config: &AppConfig,
    model_catalog: &SqliteModelCatalog,
    preview_hub: PreviewHub,
    crosshair_hub: Option<CrosshairHub>,
    parser_library: &Path,
    model_identity_cache: &ModelIdentityCache,
) -> Result<DeepStreamSessionConfig, LivePerceptionError> {
    let adapters = config
        .require_vision_adapters()
        .map_err(LivePerceptionError::Config)?;
    let capture = resolve_capture_plan(adapters.capture)?;
    let format = parse_capture_format(&capture.pixel_format)?;
    let io_mode = u32::try_from(adapters.inference.deepstream_io_mode)
        .map_err(|_| LivePerceptionError::InvalidIoMode(adapters.inference.deepstream_io_mode))?;
    let (path, model_contract) =
        resolve_active_nvinfer_config(config, model_catalog, parser_library, model_identity_cache)?;
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
    model_identity_cache: &ModelIdentityCache,
) -> Result<(PathBuf, PerceptionModelContract), LivePerceptionError> {
    let model = active_runtime_model(model_catalog)?;
    resolve_model_nvinfer_config(config, &model, None, parser_library, model_identity_cache)
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
    model_identity_cache: &ModelIdentityCache,
) -> Result<(PathBuf, PerceptionModelContract), LivePerceptionError> {
    let candidate_preflight = requested_preset.is_some();
    let inference = config
        .require_inference_adapter()
        .map_err(LivePerceptionError::Config)?;
    let parsed_manifest = validate_runtime_model(config, model, model_identity_cache)?;
    let manifest = &parsed_manifest.document;
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
    let parser =
        deepstream_parser_contract(manifest).map_err(|error| manifest_error(error.to_string()))?;
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
            output_family: if manifest.output.has_objectness {
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
    model_identity_cache: &ModelIdentityCache,
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
        model_identity_cache,
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
    render_deepstream_nvinfer_config(
        manifest,
        engine_value,
        parser_value,
        component_id,
        confidence_threshold,
        nms_threshold,
    )
    .map_err(|error| manifest_error(error.to_string()))
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
    model_identity_cache: &ModelIdentityCache,
) -> Result<String, LivePerceptionError> {
    validate_model_runtime_contract(document, output_class_names_present)
        .map_err(|error| manifest_error(error.to_string()))?;
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
    let actual_sha = model_identity_cache.sha256(engine)?;
    if actual_sha != document.artifact.sha256 {
        return Err(manifest_error(format!(
            "engine sha256 {actual_sha} does not match manifest {}",
            document.artifact.sha256
        )));
    }
    Ok(actual_sha)
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
    }

    #[test]
    fn fingerprint_matches_python_small_fixed_float_format() {
        let mut manifest = canonical_manifest();
        manifest.output.anchors = vec![vec![1e-5]];
        assert_eq!(
            compute_model_fingerprint(&manifest).expect("fingerprint"),
            "efdfa2dc09af968ea57a7a7bcbf9c098a7e83335897e8b2e95574dfbbdeb5696"
        );
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
        let legacy = "e10aff7d0417c57aed26c2bf1c8f405b5f22cecade74ec2df3515cbbe0c47ae6";
        assert!(
            model_fingerprint_matches(&canonical_manifest(), legacy, false)
                .expect("legacy fingerprint")
        );
        assert!(
            !model_fingerprint_matches(&canonical_manifest(), legacy, true)
                .expect("explicit class names require the canonical fingerprint")
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
