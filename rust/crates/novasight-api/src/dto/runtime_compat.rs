use std::collections::BTreeMap;

use novasight_runtime::{
    AppConfig, CrosshairSnapshot, PipelineState, PreviewSnapshot, RuntimeErrorSummary,
    RuntimeSnapshot, SubsystemSnapshot, SubsystemState,
};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CompatibilityHealth {
    pub ok: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CompatibilityRuntimeStart {
    pub running: bool,
    pub accepted: bool,
    pub failed: bool,
    pub epoch: Option<u64>,
    pub operation_id: Option<String>,
}

impl From<&RuntimeSnapshot> for CompatibilityRuntimeStart {
    fn from(snapshot: &RuntimeSnapshot) -> Self {
        Self {
            running: snapshot.pipeline.state == PipelineState::Running,
            accepted: snapshot.pipeline.state == PipelineState::Running,
            failed: snapshot.pipeline.state == PipelineState::Faulted,
            epoch: snapshot.pipeline.epoch.map(|epoch| epoch.0),
            operation_id: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CompatibilityStatusFrame {
    pub kind: &'static str,
    pub topic: String,
    pub full: bool,
    pub state: CompatibilityRuntimeState,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CompatibilityRuntimeState {
    pub running: bool,
    pub source: String,
    pub active_model: Option<serde_json::Value>,
    pub model_catalog_error: Option<String>,
    pub executor: ExecutorState,
    pub capture: CaptureState,
    pub statistics: StatisticsState,
    pub inference: InferenceState,
    pub config: ConfigSummary,
    pub pipeline: PipelineSummary,
    pub vision: VisionState,
    pub fatal_error: Option<RuntimeErrorSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ExecutorState {
    pub selected: String,
    pub executors: BTreeMap<String, Availability>,
    pub state: SubsystemState,
    pub last_error: Option<RuntimeErrorSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Availability {
    pub available: bool,
    pub connected: bool,
    pub connecting: bool,
    pub monitoring: bool,
    pub connection_state: &'static str,
    pub retryable: bool,
    pub last_error: Option<String>,
    pub managed_by_runtime: bool,
    pub move_count: u64,
    pub last_dx: Option<i32>,
    pub last_dy: Option<i32>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CaptureState {
    pub available: bool,
    pub running: bool,
    pub state: SubsystemState,
    pub device: String,
    pub backend: Option<String>,
    pub profile: Option<CaptureProfile>,
    pub last_error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CaptureProfile {
    pub pixel_format: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub preference: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StatisticsState {
    pub capture_counter: u64,
    pub inference_counter: u64,
    pub detection_batch_counter: u64,
    pub dropped_counter: u64,
    pub metrics_available: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct InferenceState {
    pub available: bool,
    pub configured: bool,
    pub running: bool,
    pub state: SubsystemState,
    pub selected: Option<String>,
    pub reason: Option<String>,
    pub preview_enabled: bool,
    pub preview_active: bool,
    pub preview_encoder_active: bool,
    pub preview_consumers: usize,
    pub preview_available: bool,
    pub preview_sequence: u64,
    pub preview_reason: String,
    pub preview_transport: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ConfigSummary {
    pub version: u64,
    pub schema_version: u32,
    pub effective_version: u64,
    pub restart_required: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PipelineSummary {
    pub running: bool,
    pub state: PipelineState,
    pub epoch: Option<u64>,
    pub started_at_ms: Option<u64>,
    pub mode: String,
    pub last_error: Option<RuntimeErrorSummary>,
    pub deepstream: DeepStreamState,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct DeepStreamState {
    pub crosshair_active: bool,
    pub crosshair_reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct VisionState {
    pub crosshair: Option<CrosshairSnapshot>,
}

impl CompatibilityRuntimeState {
    pub fn new(
        snapshot: &RuntimeSnapshot,
        config: Option<&AppConfig>,
        effective_revision: Option<u64>,
        hardware_output_enabled: bool,
        preview: Option<&PreviewSnapshot>,
        crosshair: Option<&CrosshairSnapshot>,
    ) -> Self {
        let capture_config = config.and_then(|config| config.capture.as_ref());
        let inference_config = config.and_then(|config| config.inference.as_ref());
        let device_config = config.and_then(|config| config.device.as_ref());
        let running = snapshot.pipeline.state == PipelineState::Running;
        let source = capture_config
            .map(|capture| capture.device.to_string_lossy().into_owned())
            .or_else(|| {
                config
                    .filter(|config| config.replay.enabled)
                    .map(|_| "replay".to_owned())
            })
            .unwrap_or_else(|| "unconfigured".to_owned());
        let capture_available =
            capture_config.is_some() && subsystem_can_run(&snapshot.subsystems.capture);
        let inference_available =
            inference_config.is_some() && subsystem_can_run(&snapshot.subsystems.inference);
        let device_available =
            device_config.is_some() && subsystem_can_run(&snapshot.subsystems.device);
        let selected_device = device_config
            .map(|device| serialized_label(&device.backend))
            .unwrap_or_else(|| "unconfigured".to_owned());
        let mut executors = BTreeMap::new();
        executors.insert(
            "dry_run".to_owned(),
            Availability {
                available: config.is_some_and(|config| config.replay.enabled),
                connected: false,
                connecting: false,
                monitoring: false,
                connection_state: "not_applicable",
                retryable: false,
                last_error: None,
                managed_by_runtime: true,
                move_count: 0,
                last_dx: None,
                last_dy: None,
            },
        );
        let device_state = snapshot.subsystems.device.state;
        let device_connected = hardware_output_enabled
            && matches!(
                device_state,
                SubsystemState::Ready | SubsystemState::Running
            );
        executors.insert(
            "kmnet".to_owned(),
            Availability {
                available: hardware_output_enabled && device_available,
                connected: device_connected,
                connecting: hardware_output_enabled && device_state == SubsystemState::Starting,
                monitoring: device_connected && running,
                connection_state: match device_state {
                    SubsystemState::Starting => "connecting",
                    SubsystemState::Ready | SubsystemState::Running => "connected",
                    SubsystemState::Failed | SubsystemState::Unavailable => "failed",
                    SubsystemState::Degraded => "degraded",
                    SubsystemState::Stopping => "disconnecting",
                    SubsystemState::Stopped => "stopped",
                },
                retryable: hardware_output_enabled
                    && matches!(
                        device_state,
                        SubsystemState::Failed | SubsystemState::Unavailable
                    ),
                last_error: snapshot
                    .subsystems
                    .device
                    .last_error
                    .as_ref()
                    .map(|error| error.message.clone()),
                managed_by_runtime: true,
                move_count: snapshot.device_metrics.diagnostic_move_count,
                last_dx: snapshot.device_metrics.last_diagnostic_dx,
                last_dy: snapshot.device_metrics.last_diagnostic_dy,
            },
        );
        let metrics = snapshot.perception_metrics;
        let dropped_counter = metrics
            .busy_dropped_batches
            .saturating_add(metrics.overwritten_snapshots)
            .saturating_add(metrics.unavailable_snapshot_slots)
            .saturating_add(metrics.extraction_rejections)
            .saturating_add(metrics.admission_rejections)
            .saturating_add(metrics.ingress_rejections)
            .saturating_add(snapshot.pipeline_metrics.input_overwrites)
            .saturating_add(snapshot.pipeline_metrics.command_overwrites)
            .saturating_add(snapshot.pipeline_metrics.superseded_commands)
            .saturating_add(snapshot.pipeline_metrics.stale_commands);
        let fatal_error = snapshot
            .pipeline
            .last_error
            .clone()
            .or_else(|| first_subsystem_error(snapshot));
        let mode = if config.is_some_and(|config| config.replay.enabled) {
            "replay".to_owned()
        } else {
            capture_config
                .map(|capture| serialized_label(&capture.backend))
                .unwrap_or_else(|| "unconfigured".to_owned())
        };

        Self {
            running,
            source,
            active_model: snapshot.model.active.as_ref().map(|active| {
                serde_json::to_value(active)
                    .expect("active model catalog records must have a JSON representation")
            }),
            model_catalog_error: snapshot.model.catalog_error.clone(),
            executor: ExecutorState {
                selected: selected_device,
                executors,
                state: snapshot.subsystems.device.state,
                last_error: snapshot.subsystems.device.last_error.clone(),
            },
            capture: CaptureState {
                available: capture_available,
                running: running && snapshot.subsystems.capture.state == SubsystemState::Running,
                state: snapshot.subsystems.capture.state,
                device: capture_config
                    .map(|capture| capture.device.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                backend: capture_config.map(|capture| serialized_label(&capture.backend)),
                profile: capture_config.map(|capture| CaptureProfile {
                    pixel_format: capture.pixel_format.clone(),
                    width: capture.width,
                    height: capture.height,
                    fps: capture.fps,
                    preference: serialized_label(&capture.preference),
                }),
                last_error: snapshot
                    .subsystems
                    .capture
                    .last_error
                    .as_ref()
                    .map(|error| error.message.clone()),
            },
            statistics: StatisticsState {
                capture_counter: metrics.probed_buffers,
                inference_counter: metrics.published_batches,
                detection_batch_counter: metrics.published_batches,
                dropped_counter,
                metrics_available: running,
            },
            inference: InferenceState {
                available: inference_available,
                configured: inference_config.is_some(),
                running: running && snapshot.subsystems.inference.state == SubsystemState::Running,
                state: snapshot.subsystems.inference.state,
                selected: inference_config.map(|inference| serialized_label(&inference.backend)),
                reason: snapshot
                    .subsystems
                    .inference
                    .last_error
                    .as_ref()
                    .map(|error| error.message.clone()),
                preview_enabled: preview.is_some_and(|preview| preview.enabled),
                preview_active: preview.is_some_and(|preview| preview.active),
                preview_encoder_active: preview.is_some_and(|preview| preview.encoder_active),
                preview_consumers: preview.map_or(0, |preview| preview.consumers),
                preview_available: preview.is_some_and(|preview| preview.available),
                preview_sequence: preview.map_or(0, |preview| preview.sequence),
                preview_reason: preview
                    .map(|preview| preview.reason.clone())
                    .unwrap_or_else(|| "hardware preview is unavailable".to_owned()),
                preview_transport: preview.map(|preview| preview.transport.clone()),
            },
            config: ConfigSummary {
                version: config.map_or(0, |config| config.revision),
                schema_version: config.map_or(0, |config| config.schema_version),
                effective_version: effective_revision.unwrap_or(0),
                restart_required: config
                    .zip(effective_revision)
                    .is_some_and(|(config, effective)| config.revision != effective),
            },
            pipeline: PipelineSummary {
                running,
                state: snapshot.pipeline.state,
                epoch: snapshot.pipeline.epoch.map(|epoch| epoch.0),
                started_at_ms: snapshot.pipeline.started_at_ms,
                mode,
                last_error: snapshot.pipeline.last_error.clone(),
                deepstream: DeepStreamState {
                    crosshair_active: crosshair.is_some_and(|state| state.running),
                    crosshair_reason: match crosshair {
                        Some(state) if state.running => String::new(),
                        Some(state) if !state.enabled => {
                            "crosshair observer is disabled by configuration".to_owned()
                        }
                        Some(_) if !running => "DeepStream pipeline is not running".to_owned(),
                        Some(_) => "crosshair observer is not running".to_owned(),
                        None => "crosshair observer is unavailable".to_owned(),
                    },
                },
            },
            vision: VisionState {
                crosshair: crosshair.cloned(),
            },
            fatal_error,
        }
    }
}

fn subsystem_can_run(subsystem: &SubsystemSnapshot) -> bool {
    !matches!(
        subsystem.state,
        SubsystemState::Failed | SubsystemState::Unavailable
    )
}

fn first_subsystem_error(snapshot: &RuntimeSnapshot) -> Option<RuntimeErrorSummary> {
    [
        &snapshot.subsystems.capture,
        &snapshot.subsystems.inference,
        &snapshot.subsystems.control,
        &snapshot.subsystems.device,
    ]
    .into_iter()
    .find_map(|subsystem| subsystem.last_error.clone())
}

fn serialized_label(value: &impl Serialize) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned())
}
