use std::collections::BTreeMap;

use novasight_runtime::{
    AppConfig, PipelineState, RuntimeErrorSummary, RuntimeSnapshot, SubsystemSnapshot,
    SubsystemState,
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
    pub executor: ExecutorState,
    pub capture: CaptureState,
    pub statistics: StatisticsState,
    pub inference: InferenceState,
    pub config: ConfigSummary,
    pub pipeline: PipelineSummary,
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
}

impl CompatibilityRuntimeState {
    pub fn new(
        snapshot: &RuntimeSnapshot,
        config: Option<&AppConfig>,
        effective_revision: Option<u64>,
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
            },
        );
        executors.insert(
            "kmnet".to_owned(),
            Availability {
                available: device_available,
            },
        );
        let metrics = snapshot.perception_metrics;
        let dropped_counter = metrics
            .busy_dropped_batches
            .saturating_add(metrics.overwritten_snapshots)
            .saturating_add(metrics.unavailable_snapshot_slots)
            .saturating_add(metrics.extraction_rejections)
            .saturating_add(metrics.admission_rejections)
            .saturating_add(metrics.ingress_rejections);
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
            active_model: None,
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
