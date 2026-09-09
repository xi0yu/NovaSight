use std::collections::BTreeMap;

use novasight_core::controller::recoil::{RecoilBlockReason, RecoilState};
use novasight_core::controller::{BlockReason, ControlMode};
use novasight_core::prediction::PredictionMotionState;
use novasight_core::tracking::{LockReason, TargetSelection};
use novasight_runtime::{
    AppConfig, CrosshairSnapshot, DetectionTelemetryItem, OutputDeliveryState, PipelineState,
    PreviewSnapshot, RuntimeErrorSummary, RuntimeSnapshot, SubsystemSnapshot, SubsystemState,
};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeHealth {
    pub ok: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeStatusFrame<T> {
    pub kind: &'static str,
    pub topic: &'static str,
    pub full: bool,
    pub state: T,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeStatusState {
    pub semantic: RuntimeSemanticState,
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
pub(crate) struct RuntimeSemanticState {
    pub daemon_instance_id: String,
    pub phase: &'static str,
    pub perception_phase: &'static str,
    pub epoch: Option<u64>,
    pub snapshot_sequence: u64,
    pub snapshot_updated_at_ms: u64,
}

#[derive(Serialize)]
struct RuntimeStatusPatch<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    semantic: Option<&'a RuntimeSemanticState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    running: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    active_model: Option<&'a Option<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_catalog_error: Option<&'a Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    executor: Option<&'a ExecutorState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capture: Option<&'a CaptureState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    statistics: Option<&'a StatisticsState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inference: Option<&'a InferenceState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<&'a ConfigSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pipeline: Option<&'a PipelineSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vision: Option<&'a VisionState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fatal_error: Option<&'a Option<RuntimeErrorSummary>>,
}

impl RuntimeStatusPatch<'_> {
    fn empty() -> Self {
        Self {
            semantic: None,
            running: None,
            source: None,
            active_model: None,
            model_catalog_error: None,
            executor: None,
            capture: None,
            statistics: None,
            inference: None,
            config: None,
            pipeline: None,
            vision: None,
            fatal_error: None,
        }
    }
}

pub(crate) fn serialize_runtime_status_frame(
    topic: &'static str,
    full: bool,
    state: &RuntimeStatusState,
) -> serde_json::Result<String> {
    if full || topic == "full" {
        return serde_json::to_string(&RuntimeStatusFrame {
            kind: "runtime_snapshot",
            topic,
            full: true,
            state,
        });
    }

    let mut patch = RuntimeStatusPatch::empty();
    // Studio keeps shared lifecycle, readiness, error, and device surfaces
    // mounted on every page. A topic therefore selects expensive detail while
    // every frame still owns a complete top-level runtime state; omitting a
    // section here would preserve stale data in the frontend merge.
    patch.semantic = Some(&state.semantic);
    patch.running = Some(state.running);
    patch.source = Some(&state.source);
    patch.active_model = Some(&state.active_model);
    patch.model_catalog_error = Some(&state.model_catalog_error);
    patch.executor = Some(&state.executor);
    patch.capture = Some(&state.capture);
    patch.statistics = Some(&state.statistics);
    patch.inference = Some(&state.inference);
    patch.config = Some(&state.config);
    patch.pipeline = Some(&state.pipeline);
    patch.vision = Some(&state.vision);
    patch.fatal_error = Some(&state.fatal_error);
    serde_json::to_string(&RuntimeStatusFrame {
        kind: "runtime_snapshot",
        topic,
        full: false,
        state: patch,
    })
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
    pub configuration_state: &'static str,
    pub configuration_ready: bool,
    pub restart_required: bool,
    pub can_connect: bool,
    pub can_disconnect: bool,
    pub blocked_reason: Option<&'static str>,
    pub connected: bool,
    pub runtime_connected: bool,
    pub connecting: bool,
    pub buttons_available: bool,
    pub button_left: bool,
    pub button_right: bool,
    pub connection_state: &'static str,
    pub retryable: bool,
    pub last_error: Option<String>,
    pub managed_by_runtime: bool,
    pub accepted_command_count: u64,
    pub last_accepted_dx: Option<i32>,
    pub last_accepted_dy: Option<i32>,
    pub diagnostic_move_count: u64,
    pub last_diagnostic_dx: Option<i32>,
    pub last_diagnostic_dy: Option<i32>,
    pub device_error_count: u64,
    pub device_recovery_count: u64,
    pub last_device_error: Option<String>,
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
    pub source: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StatisticsState {
    pub nvinfer_input_counter: u64,
    pub detection_batch_counter: u64,
    pub detection_batch_consumed_counter: u64,
    pub targeting_batch_counter: u64,
    pub nvinfer_input_fps: Option<f64>,
    pub nvinfer_output_fps: Option<f64>,
    pub detection_batch_fps: Option<f64>,
    pub targeting_batch_fps: Option<f64>,
    pub detection_data_age_ms: Option<f64>,
    pub detection_freshness_threshold_ms: Option<f64>,
    pub inference_latency_ms: Option<f64>,
    pub inference_latency_samples: u64,
    pub telemetry_window_ms: Option<u64>,
    pub metrics_available: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct InferenceState {
    pub available: bool,
    pub configured: bool,
    pub loaded: bool,
    pub running: bool,
    pub terminal_error: bool,
    pub state: SubsystemState,
    pub selected: Option<String>,
    pub reason: Option<String>,
    pub detail: Option<String>,
    pub inference_reason: Option<String>,
    pub input_frames: u64,
    pub output_buffers: u64,
    pub metadata_extractions: u64,
    pub published_batches: u64,
    pub timestamp_buffer_pts_matches: u64,
    pub timestamp_frame_meta_pts_matches: u64,
    pub timestamp_correlation_misses: u64,
    pub sampled_detection_generation: Option<u64>,
    pub preview_enabled: bool,
    pub preview_active: bool,
    pub preview_encoder_active: bool,
    pub preview_consumers: usize,
    pub preview_available: bool,
    pub preview_sequence: u64,
    pub preview_reason: String,
    pub preview_transport: Option<String>,
    pub postprocess: Option<PostprocessState>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PostprocessState {
    pub confidence_threshold: f64,
    pub nms_threshold: f64,
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
    pub running: bool,
    pub terminal_error: bool,
    pub last_error: Option<String>,
    pub input_frames: u64,
    pub metadata_extractions: u64,
    pub published_batches: u64,
    pub crosshair_active: bool,
    pub crosshair_reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct VisionState {
    pub crosshair: Option<CrosshairSnapshot>,
    pub inference: VisionInferenceState,
    pub detections: usize,
    pub detection_items: Vec<VisionDetectionState>,
    pub detection_items_truncated: usize,
    pub target: Option<VisionTargetState>,
    pub target_pipeline: TargetPipelineState,
    pub output_trace: OutputTraceState,
    pub control: VisionControlState,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OutputTraceState {
    pub code: &'static str,
    pub state: &'static str,
    pub detail: &'static str,
    pub next_action: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct VisionInferenceState {
    /// Coordinate space of the latest admitted detection batch. These remain
    /// absent until a real batch arrives, so clients can fall back explicitly.
    pub input_width: Option<u32>,
    pub input_height: Option<u32>,
    pub generation: Option<u64>,
    pub model_input_width: Option<u32>,
    pub model_input_height: Option<u32>,
    pub source_width: Option<u32>,
    pub source_height: Option<u32>,
    pub roi_offset_x: Option<u32>,
    pub roi_offset_y: Option<u32>,
    pub roi_width: Option<u32>,
    pub roi_height: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct VisionDetectionState {
    pub object_id: u64,
    pub class_id: u32,
    pub cls: u32,
    pub score: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub cx: f64,
    pub cy: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct VisionTargetState {
    pub target_detection_index: Option<usize>,
    pub track_id: u64,
    pub class_id: u32,
    pub cls: u32,
    pub score: f32,
    pub identity_confidence: f64,
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub box_cx: f64,
    pub box_cy: f64,
    pub cx: f64,
    pub cy: f64,
    pub observed_aim_x: f64,
    pub observed_aim_y: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TargetPipelineState {
    pub code: &'static str,
    pub stage: &'static str,
    pub message: &'static str,
    pub rejection_reasons: Vec<&'static str>,
    pub counts: TargetPipelineCounts,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TargetPipelineCounts {
    pub raw_candidates: usize,
    pub eligible_candidates: usize,
    pub selected_targets: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct VisionControlState {
    pub global_state: &'static str,
    pub output_enabled: bool,
    pub aim_x: Option<f64>,
    pub aim_y: Option<f64>,
    pub dx: Option<i32>,
    pub dy: Option<i32>,
    pub will_emit: Option<bool>,
    pub trigger_active: Option<bool>,
    pub reason: Option<&'static str>,
    pub no_send_reason: Option<&'static str>,
    pub candidates: usize,
    pub selector_state: &'static str,
    pub selection_reason: Option<&'static str>,
    pub candidate_filter: CandidateFilterState,
    pub mouse_observation: MouseObservationState,
    pub pipeline: ControlPipelineState,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct CandidateFilterState {
    pub effective_class_filter: String,
    pub basic: BasicCandidateFilterState,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BasicCandidateFilterState {
    pub raw_candidates: usize,
    pub filtered_candidates: usize,
    pub rejected_candidates: usize,
    pub rejected_class_ids: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MouseObservationState {
    pub control_width_px: Option<u32>,
    pub control_height_px: Option<u32>,
    pub observed_x_px: Option<f64>,
    pub observed_y_px: Option<f64>,
    pub predicted_x_px: Option<f64>,
    pub predicted_y_px: Option<f64>,
    pub measurement_dt_s: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ControlPipelineState {
    pub control_mode: &'static str,
    pub movement_strategy: &'static str,
    pub output_delivery_state: OutputDeliveryState,
    pub mode: Option<&'static str>,
    pub frame_age_ms: Option<f64>,
    pub history_position_count: Option<usize>,
    pub velocity_1: Option<f64>,
    pub velocity_2: Option<f64>,
    pub velocity_3: Option<f64>,
    pub medoid_velocity: Option<f64>,
    pub prediction_velocity: Option<f64>,
    pub motion_state: Option<PredictionMotionState>,
    pub measurement_dt_s: Option<f64>,
    pub reference_dt_ms: Option<f64>,
    pub prediction_actuation_delay_ms: Option<f64>,
    pub prediction_lead_ms: Option<f64>,
    pub prediction_horizon_ms: Option<f64>,
    pub prediction_raw_offset_x: Option<f64>,
    pub prediction_allowed_cap_x: Option<f64>,
    pub prediction_safe_offset_x: Option<f64>,
    pub prediction_allowed: Option<bool>,
    pub velocity_y_1: Option<f64>,
    pub velocity_y_2: Option<f64>,
    pub velocity_y_3: Option<f64>,
    pub medoid_velocity_y: Option<f64>,
    pub prediction_velocity_y: Option<f64>,
    pub motion_state_y: Option<PredictionMotionState>,
    pub measurement_dt_y_s: Option<f64>,
    pub reference_dt_y_ms: Option<f64>,
    pub prediction_raw_offset_y: Option<f64>,
    pub prediction_allowed_cap_y: Option<f64>,
    pub prediction_safe_offset_y: Option<f64>,
    pub prediction_allowed_y: Option<bool>,
    pub observed_error_x_px: Option<f64>,
    pub observed_error_y_px: Option<f64>,
    pub predicted_error_x_px: Option<f64>,
    pub predicted_error_y_px: Option<f64>,
    pub full_error_counts_x: Option<f64>,
    pub full_error_counts_y: Option<f64>,
    pub float_demand_x: Option<f64>,
    pub float_demand_y: Option<f64>,
    pub integer_command_x: Option<i32>,
    pub integer_command_y: Option<i32>,
    pub quantizer_residual_x: Option<f64>,
    pub quantizer_residual_y: Option<f64>,
    pub block_reason: Option<&'static str>,
    pub fire_delay_enabled: bool,
    pub fire_delay_configured_ms: u64,
    pub fire_delay_pending: bool,
    pub fire_delay_elapsed_ms: Option<f64>,
    pub fire_delay_remaining_ms: Option<f64>,
    pub recoil_mode: &'static str,
    pub recoil_enabled: bool,
    pub recoil_active: bool,
    pub recoil_state: RecoilState,
    pub recoil_interval_ms: u64,
    pub recoil_y_counts: i32,
    pub recoil_elapsed_since_output_ms: Option<f64>,
    pub recoil_remaining_ms: f64,
    pub recoil_requested_counts_y: i32,
    pub recoil_emitted_counts_y: i32,
    pub recoil_source_generation: Option<u64>,
    pub recoil_block_reason: RecoilBlockReason,
}

impl RuntimeStatusState {
    #[cfg(test)]
    pub fn new(
        snapshot: &RuntimeSnapshot,
        config: Option<&AppConfig>,
        effective_revision: Option<u64>,
        hardware_output_enabled: bool,
        preview: Option<&PreviewSnapshot>,
        crosshair: Option<&CrosshairSnapshot>,
    ) -> Self {
        Self::build(
            "test-daemon",
            snapshot,
            config,
            config,
            effective_revision,
            hardware_output_enabled,
            preview,
            crosshair,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn for_topic(
        daemon_instance_id: &str,
        snapshot: &RuntimeSnapshot,
        config: Option<&AppConfig>,
        effective_config: Option<&AppConfig>,
        effective_revision: Option<u64>,
        hardware_output_enabled: bool,
        preview: Option<&PreviewSnapshot>,
        crosshair: Option<&CrosshairSnapshot>,
        topic: &'static str,
    ) -> Self {
        Self::build(
            daemon_instance_id,
            snapshot,
            config,
            effective_config,
            effective_revision,
            hardware_output_enabled,
            preview,
            crosshair,
            matches!(topic, "full" | "infer" | "control"),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        daemon_instance_id: &str,
        snapshot: &RuntimeSnapshot,
        config: Option<&AppConfig>,
        effective_config: Option<&AppConfig>,
        effective_revision: Option<u64>,
        hardware_output_enabled: bool,
        preview: Option<&PreviewSnapshot>,
        crosshair: Option<&CrosshairSnapshot>,
        include_detection_items: bool,
    ) -> Self {
        let runtime_config = effective_config.or(config);
        let capture_config = runtime_config.and_then(|config| config.capture.as_ref());
        let inference_config = runtime_config.and_then(|config| config.inference.as_ref());
        let model_input = snapshot
            .model
            .input_width
            .zip(snapshot.model.input_height)
            .or_else(|| {
                snapshot
                    .model
                    .active
                    .as_ref()
                    .and_then(|active| parse_nchw_dimensions(&active.version.input_shape))
            });
        let device_config = runtime_config.and_then(|config| config.device.as_ref());
        let control = snapshot.pipeline_metrics.control;
        let target_selection = &snapshot.pipeline_metrics.target_selection;
        let has_target_sample = snapshot.pipeline_metrics.targeting_batches > 0;
        let target_detection_index = target_selection.target_object_id.and_then(|object_id| {
            snapshot
                .pipeline_metrics
                .detections
                .items
                .iter()
                .position(|detection| detection.object_id == object_id)
        });
        let target = vision_target_state(target_selection, target_detection_index);
        let target_pipeline = target_pipeline_state(target_selection, has_target_sample);
        let effective_class_filter = config
            .map(|config| config.pipeline.target_class_filter.clone())
            .unwrap_or_else(|| "all".to_owned());
        let control_sample = control.sample_available;
        let trigger_delay_pending = snapshot.pipeline_metrics.trigger_delay.pending;
        let control_reason = if trigger_delay_pending {
            Some("TRIGGER_DELAY_PENDING")
        } else {
            control_sample.then_some(block_reason_label(control.block_reason))
        };
        let hardware_trigger_required = config.is_some_and(|config| {
            matches!(
                config.control.trigger_mode,
                novasight_store::config::TriggerMode::Hardware
            )
        });
        let observed_trigger_active = if snapshot.pipeline.state != PipelineState::Running {
            None
        } else if hardware_trigger_required {
            snapshot.pipeline_metrics.buttons_available.then_some(
                snapshot.pipeline_metrics.button_left || snapshot.pipeline_metrics.button_right,
            )
        } else {
            Some(true)
        };
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
        let config_restart_required = config
            .zip(effective_revision)
            .is_some_and(|(config, effective)| config.revision != effective);
        let hardware_restart_required = config_restart_required
            && serde_json::to_value(config.and_then(|config| config.device.as_ref())).ok()
                != serde_json::to_value(runtime_config.and_then(|config| config.device.as_ref()))
                    .ok();
        let selected_device = device_config
            .map(|device| serialized_label(&device.backend))
            .unwrap_or_else(|| "unconfigured".to_owned());
        let mut executors = BTreeMap::new();
        executors.insert(
            "replay".to_owned(),
            Availability {
                available: config.is_some_and(|config| config.replay.enabled),
                configuration_state: "not_applicable",
                configuration_ready: true,
                restart_required: false,
                can_connect: false,
                can_disconnect: false,
                blocked_reason: None,
                connected: false,
                runtime_connected: false,
                connecting: false,
                buttons_available: false,
                button_left: false,
                button_right: false,
                connection_state: "not_applicable",
                retryable: false,
                last_error: None,
                managed_by_runtime: true,
                accepted_command_count: 0,
                last_accepted_dx: None,
                last_accepted_dy: None,
                diagnostic_move_count: 0,
                last_diagnostic_dx: None,
                last_diagnostic_dy: None,
                device_error_count: 0,
                device_recovery_count: 0,
                last_device_error: None,
            },
        );
        let device_state = snapshot.subsystems.device.state;
        let device_uncommissioned = snapshot
            .subsystems
            .device
            .last_error
            .as_ref()
            .is_some_and(|error| error.code == "device_uncommissioned");
        let device_connected = hardware_output_enabled
            && matches!(
                device_state,
                SubsystemState::Ready | SubsystemState::Running
            );
        let device_configuration_ready =
            hardware_output_enabled && !device_uncommissioned && !hardware_restart_required;
        let can_connect = device_configuration_ready
            && running
            && !snapshot.pipeline_metrics.device_connected
            && device_state != SubsystemState::Starting;
        // Disconnect is safety-monotonic and remains admissible even when a
        // newer desired configuration is waiting for daemon restart.
        let can_disconnect = hardware_output_enabled
            && running
            && snapshot.pipeline_metrics.device_connection_enabled;
        let blocked_reason = if !hardware_output_enabled {
            Some("hardware_output_disabled")
        } else if hardware_restart_required {
            Some("daemon_restart_required")
        } else if device_uncommissioned {
            Some("device_uncommissioned")
        } else if !running {
            Some("runtime_stopped")
        } else if snapshot.pipeline_metrics.device_connected {
            Some("already_connected")
        } else if device_state == SubsystemState::Starting {
            Some("connecting")
        } else {
            None
        };
        executors.insert(
            "kmnet".to_owned(),
            Availability {
                available: hardware_output_enabled && device_available,
                configuration_state: if hardware_restart_required {
                    "restart_required"
                } else if device_uncommissioned {
                    "uncommissioned"
                } else {
                    "ready"
                },
                configuration_ready: device_configuration_ready,
                restart_required: hardware_restart_required,
                can_connect,
                can_disconnect,
                blocked_reason,
                connected: device_connected,
                runtime_connected: hardware_output_enabled
                    && snapshot.pipeline_metrics.device_connected,
                connecting: hardware_output_enabled && device_state == SubsystemState::Starting,
                buttons_available: snapshot.pipeline_metrics.buttons_available,
                button_left: snapshot.pipeline_metrics.button_left,
                button_right: snapshot.pipeline_metrics.button_right,
                connection_state: if device_uncommissioned {
                    "uncommissioned"
                } else {
                    match device_state {
                        SubsystemState::Starting => "connecting",
                        SubsystemState::Ready | SubsystemState::Running => "connected",
                        SubsystemState::Failed | SubsystemState::Unavailable => "failed",
                        SubsystemState::Degraded => "degraded",
                        SubsystemState::Stopping => "disconnecting",
                        SubsystemState::Stopped => "stopped",
                    }
                },
                retryable: !device_uncommissioned
                    && hardware_output_enabled
                    && matches!(
                        device_state,
                        SubsystemState::Degraded
                            | SubsystemState::Failed
                            | SubsystemState::Unavailable
                    ),
                last_error: snapshot
                    .subsystems
                    .device
                    .last_error
                    .as_ref()
                    .map(|error| error.message.clone()),
                managed_by_runtime: true,
                accepted_command_count: snapshot.pipeline_metrics.device_receipts,
                last_accepted_dx: snapshot
                    .pipeline_metrics
                    .last_device_receipt
                    .map(|receipt| receipt.delta_x_counts),
                last_accepted_dy: snapshot
                    .pipeline_metrics
                    .last_device_receipt
                    .map(|receipt| receipt.delta_y_counts),
                diagnostic_move_count: snapshot.device_metrics.diagnostic_move_count,
                last_diagnostic_dx: snapshot.device_metrics.last_diagnostic_dx,
                last_diagnostic_dy: snapshot.device_metrics.last_diagnostic_dy,
                device_error_count: snapshot.pipeline_metrics.device_error_count,
                device_recovery_count: snapshot.pipeline_metrics.device_recovery_count,
                last_device_error: snapshot.pipeline_metrics.last_device_error.clone(),
            },
        );
        let metrics = snapshot.perception_metrics;
        let inference_running =
            running && snapshot.subsystems.inference.state == SubsystemState::Running;
        let inference_terminal_error =
            snapshot.subsystems.inference.state == SubsystemState::Failed;
        let inference_error = snapshot
            .subsystems
            .inference
            .last_error
            .as_ref()
            .map(|error| error.message.clone());
        let output_trace = OutputTraceProjection {
            snapshot,
            runtime_config,
            target_pipeline: &target_pipeline,
            hardware_output_enabled,
            control_reason,
        }
        .state();
        let metadata_extractions = metrics
            .probed_buffers
            .saturating_sub(metrics.unavailable_snapshot_slots)
            .saturating_sub(metrics.extraction_rejections);
        let fatal_error = snapshot
            .pipeline
            .last_error
            .clone()
            .or_else(|| first_subsystem_error(snapshot, hardware_output_enabled));
        let mode = if config.is_some_and(|config| config.replay.enabled) {
            "replay".to_owned()
        } else {
            capture_config
                .map(|capture| serialized_label(&capture.backend))
                .unwrap_or_else(|| "unconfigured".to_owned())
        };
        let waiting_model = running
            && inference_config.is_some()
            && snapshot.model.active.is_none()
            && snapshot.subsystems.inference.state != SubsystemState::Unavailable;
        let semantic_phase = match snapshot.pipeline.state {
            PipelineState::Running if waiting_model => "waiting_model",
            PipelineState::Running => "running",
            PipelineState::Starting => "starting",
            PipelineState::Stopping => "stopping",
            PipelineState::Faulted => "faulted",
            PipelineState::Standby => "standby",
            PipelineState::Stopped => "stopped",
        };
        let perception_phase = if waiting_model {
            "waiting_model"
        } else if snapshot.subsystems.inference.state == SubsystemState::Running {
            "running"
        } else if snapshot.subsystems.inference.state == SubsystemState::Starting {
            "starting"
        } else if snapshot.subsystems.inference.state == SubsystemState::Failed {
            "faulted"
        } else if snapshot.subsystems.inference.state == SubsystemState::Unavailable {
            "unavailable"
        } else if inference_config.is_some() {
            "stopped"
        } else {
            "unavailable"
        };

        Self {
            semantic: RuntimeSemanticState {
                daemon_instance_id: daemon_instance_id.to_owned(),
                phase: semantic_phase,
                perception_phase,
                epoch: snapshot.pipeline.epoch.map(|epoch| epoch.0),
                snapshot_sequence: snapshot.sequence,
                snapshot_updated_at_ms: snapshot.updated_at_ms,
            },
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
                    source: "configured",
                }),
                last_error: snapshot
                    .subsystems
                    .capture
                    .last_error
                    .as_ref()
                    .map(|error| error.message.clone()),
            },
            statistics: StatisticsState {
                nvinfer_input_counter: metrics.input_buffers,
                detection_batch_counter: metrics.published_batches,
                detection_batch_consumed_counter: snapshot.pipeline_metrics.received_batches,
                targeting_batch_counter: snapshot.pipeline_metrics.targeting_batches,
                nvinfer_input_fps: snapshot.telemetry.nvinfer_input_fps,
                nvinfer_output_fps: snapshot.telemetry.nvinfer_output_fps,
                detection_batch_fps: snapshot.telemetry.detection_batch_fps,
                targeting_batch_fps: snapshot.telemetry.targeting_batch_fps,
                detection_data_age_ms: snapshot.telemetry.detection_data_age_ms,
                detection_freshness_threshold_ms: runtime_config
                    .map(|config| config.pipeline.freshness_threshold_ms),
                inference_latency_ms: snapshot.telemetry.inference_latency_ms,
                inference_latency_samples: metrics.inference_duration_samples,
                telemetry_window_ms: snapshot.telemetry.sample_window_ms,
                metrics_available: metrics.input_buffers > 0
                    || metrics.probed_buffers > 0
                    || metrics.published_batches > 0
                    || snapshot.pipeline_metrics.received_batches > 0,
            },
            inference: InferenceState {
                available: inference_available,
                configured: inference_config.is_some(),
                loaded: inference_running,
                running: inference_running,
                terminal_error: inference_terminal_error,
                state: snapshot.subsystems.inference.state,
                selected: inference_config.map(|inference| serialized_label(&inference.backend)),
                reason: inference_error.clone(),
                detail: inference_error.clone(),
                inference_reason: inference_error.clone(),
                input_frames: metrics.input_buffers,
                output_buffers: metrics.probed_buffers,
                metadata_extractions,
                published_batches: metrics.published_batches,
                timestamp_buffer_pts_matches: metrics.timestamp_buffer_pts_matches,
                timestamp_frame_meta_pts_matches: metrics.timestamp_frame_meta_pts_matches,
                timestamp_correlation_misses: metrics.timestamp_correlation_misses,
                sampled_detection_generation: snapshot
                    .pipeline_metrics
                    .detections
                    .generation
                    .map(|generation| generation.0),
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
                postprocess: inference_config.map(|inference| PostprocessState {
                    confidence_threshold: inference.confidence_threshold,
                    nms_threshold: inference.nms_threshold,
                }),
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
                    running: inference_running,
                    terminal_error: inference_terminal_error,
                    last_error: inference_error,
                    input_frames: metrics.input_buffers,
                    metadata_extractions,
                    published_batches: metrics.published_batches,
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
                inference: VisionInferenceState {
                    input_width: snapshot
                        .pipeline_metrics
                        .detections
                        .generation
                        .map(|_| snapshot.pipeline_metrics.detections.coordinate_width),
                    input_height: snapshot
                        .pipeline_metrics
                        .detections
                        .generation
                        .map(|_| snapshot.pipeline_metrics.detections.coordinate_height),
                    generation: snapshot
                        .pipeline_metrics
                        .detections
                        .generation
                        .map(|generation| generation.0),
                    model_input_width: model_input.map(|(width, _)| width),
                    model_input_height: model_input.map(|(_, height)| height),
                    source_width: capture_config.map(|config| config.width),
                    source_height: capture_config.map(|config| config.height),
                    roi_offset_x: capture_config.map(|config| config.roi_left),
                    roi_offset_y: capture_config.map(|config| config.roi_top),
                    roi_width: capture_config.map(|config| config.roi_width),
                    roi_height: capture_config.map(|config| config.roi_height),
                },
                detections: target_selection.candidates,
                detection_items: if include_detection_items {
                    snapshot
                        .pipeline_metrics
                        .detections
                        .items
                        .iter()
                        .map(vision_detection_state)
                        .collect()
                } else {
                    Vec::new()
                },
                detection_items_truncated: if include_detection_items {
                    snapshot.pipeline_metrics.detections.truncated
                } else {
                    0
                },
                target,
                target_pipeline,
                output_trace,
                control: VisionControlState {
                    global_state: if trigger_delay_pending {
                        "WAITING_TRIGGER_DELAY"
                    } else if control_sample {
                        "CALCULATED"
                    } else {
                        "IDLE"
                    },
                    output_enabled: snapshot.pipeline_metrics.output_gate_open,
                    aim_x: control_sample.then_some(control.aim_x + control.predicted_offset_x),
                    aim_y: control_sample.then_some(control.aim_y + control.predicted_offset_y),
                    dx: control_sample.then_some(control.dx),
                    dy: control_sample.then_some(control.dy),
                    will_emit: if trigger_delay_pending {
                        Some(false)
                    } else {
                        control_sample.then_some(control.emit_allowed)
                    },
                    trigger_active: observed_trigger_active,
                    reason: control_reason,
                    no_send_reason: control_reason.filter(|reason| !reason.is_empty()),
                    candidates: target_selection.inside_fov,
                    selector_state: if target_selection.target_track_id.is_some() {
                        "LOCKED"
                    } else {
                        "SEARCHING"
                    },
                    selection_reason: target_selection.lock_reason.map(lock_reason_label),
                    candidate_filter: CandidateFilterState {
                        effective_class_filter,
                        basic: BasicCandidateFilterState {
                            raw_candidates: target_selection.candidates,
                            filtered_candidates: target_selection
                                .candidates
                                .saturating_sub(target_selection.rejected_by_confidence)
                                .saturating_sub(target_selection.rejected_by_class),
                            rejected_candidates: target_selection
                                .rejected_by_confidence
                                .saturating_add(target_selection.rejected_by_class),
                            rejected_class_ids: target_selection.rejected_class_ids.clone(),
                        },
                    },
                    mouse_observation: MouseObservationState {
                        control_width_px: control_sample.then_some(control.observation_width),
                        control_height_px: control_sample.then_some(control.observation_height),
                        observed_x_px: control_sample.then_some(control.aim_x),
                        observed_y_px: control_sample.then_some(control.aim_y),
                        predicted_x_px: control_sample
                            .then_some(control.aim_x + control.predicted_offset_x),
                        predicted_y_px: control_sample
                            .then_some(control.aim_y + control.predicted_offset_y),
                        measurement_dt_s: control.measurement_dt_ms.map(|value| value / 1_000.0),
                    },
                    pipeline: ControlPipelineState {
                        control_mode: "continuous_atan_medoid_v2",
                        movement_strategy: "latest_replace",
                        output_delivery_state: snapshot.pipeline_metrics.output_delivery_state,
                        mode: control_sample.then_some(control_mode_label(control.mode)),
                        frame_age_ms: control_sample.then_some(control.frame_age_ms),
                        history_position_count: control_sample
                            .then_some(control.history_position_count),
                        velocity_1: control.velocity_samples[0],
                        velocity_2: control.velocity_samples[1],
                        velocity_3: control.velocity_samples[2],
                        medoid_velocity: control.medoid_velocity,
                        prediction_velocity: control_sample.then_some(control.velocity_x),
                        motion_state: control_sample.then_some(control.motion_state),
                        measurement_dt_s: control.measurement_dt_ms.map(|value| value / 1_000.0),
                        reference_dt_ms: control_sample.then_some(control.reference_dt_ms),
                        prediction_actuation_delay_ms: control_sample
                            .then_some(control.prediction_actuation_delay_ms),
                        prediction_lead_ms: control_sample.then_some(control.prediction_lead_ms),
                        prediction_horizon_ms: control_sample
                            .then_some(control.prediction_horizon_ms),
                        prediction_raw_offset_x: control_sample
                            .then_some(control.prediction_raw_offset_x),
                        prediction_allowed_cap_x: control_sample
                            .then_some(control.prediction_allowed_cap_x),
                        prediction_safe_offset_x: control_sample
                            .then_some(control.predicted_offset_x),
                        prediction_allowed: control_sample.then_some(control.prediction_allowed),
                        velocity_y_1: control.velocity_samples_y[0],
                        velocity_y_2: control.velocity_samples_y[1],
                        velocity_y_3: control.velocity_samples_y[2],
                        medoid_velocity_y: control.medoid_velocity_y,
                        prediction_velocity_y: control_sample.then_some(control.velocity_y),
                        motion_state_y: control_sample.then_some(control.motion_state_y),
                        measurement_dt_y_s: control
                            .measurement_dt_ms_y
                            .map(|value| value / 1_000.0),
                        reference_dt_y_ms: control_sample.then_some(control.reference_dt_ms_y),
                        prediction_raw_offset_y: control_sample
                            .then_some(control.prediction_raw_offset_y),
                        prediction_allowed_cap_y: control_sample
                            .then_some(control.prediction_allowed_cap_y),
                        prediction_safe_offset_y: control_sample
                            .then_some(control.predicted_offset_y),
                        prediction_allowed_y: control_sample
                            .then_some(control.prediction_allowed_y),
                        observed_error_x_px: control_sample.then_some(control.observed_error_x),
                        observed_error_y_px: control_sample.then_some(control.observed_error_y),
                        predicted_error_x_px: control_sample.then_some(control.filtered_error_x),
                        predicted_error_y_px: control_sample.then_some(control.filtered_error_y),
                        full_error_counts_x: control_sample.then_some(control.full_error_counts_x),
                        full_error_counts_y: control_sample.then_some(control.full_error_counts_y),
                        float_demand_x: control_sample.then_some(control.float_demand_x),
                        float_demand_y: control_sample.then_some(control.float_demand_y),
                        integer_command_x: control_sample.then_some(control.dx),
                        integer_command_y: control_sample.then_some(control.dy),
                        quantizer_residual_x: control_sample
                            .then_some(control.quantizer_residual_x),
                        quantizer_residual_y: control_sample
                            .then_some(control.quantizer_residual_y),
                        block_reason: control_reason,
                        fire_delay_enabled: config
                            .is_some_and(|config| config.pipeline.fire_delay_enabled),
                        fire_delay_configured_ms: snapshot
                            .pipeline_metrics
                            .trigger_delay
                            .configured_ms
                            .unwrap_or_else(|| {
                                config.map_or(0, |config| config.pipeline.fire_delay_ms)
                            }),
                        fire_delay_pending: snapshot.pipeline_metrics.trigger_delay.pending,
                        fire_delay_elapsed_ms: snapshot.pipeline_metrics.trigger_delay.elapsed_ms,
                        fire_delay_remaining_ms: snapshot
                            .pipeline_metrics
                            .trigger_delay
                            .remaining_ms,
                        recoil_mode: config.map_or("interval_additive", |config| {
                            if config.control.recoil.require_target {
                                "target_guarded_interval_additive"
                            } else {
                                "interval_additive"
                            }
                        }),
                        recoil_enabled: config.is_some_and(|config| config.control.recoil.enabled),
                        recoil_active: snapshot.pipeline_metrics.recoil.engaged(),
                        recoil_state: snapshot.pipeline_metrics.recoil.state,
                        recoil_interval_ms: snapshot.pipeline_metrics.recoil.interval_ms,
                        recoil_y_counts: snapshot.pipeline_metrics.recoil.configured_y_counts,
                        recoil_elapsed_since_output_ms: snapshot
                            .pipeline_metrics
                            .recoil
                            .elapsed_since_output_ms,
                        recoil_remaining_ms: snapshot.pipeline_metrics.recoil.remaining_ms,
                        recoil_requested_counts_y: snapshot
                            .pipeline_metrics
                            .recoil
                            .requested_counts_y,
                        recoil_emitted_counts_y: snapshot.pipeline_metrics.recoil.emitted_counts_y,
                        recoil_source_generation: snapshot
                            .pipeline_metrics
                            .recoil
                            .source_generation,
                        recoil_block_reason: snapshot.pipeline_metrics.recoil.block_reason,
                    },
                },
            },
            fatal_error,
        }
    }
}

fn parse_nchw_dimensions(shape: &str) -> Option<(u32, u32)> {
    let dimensions = shape
        .split(['x', 'X', ',', ' '])
        .filter(|value| !value.is_empty())
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if dimensions.len() != 4 || dimensions[0] != 1 {
        return None;
    }
    Some((dimensions[3], dimensions[2]))
}

fn vision_detection_state(detection: &DetectionTelemetryItem) -> VisionDetectionState {
    VisionDetectionState {
        object_id: detection.object_id,
        class_id: detection.class_id,
        cls: detection.class_id,
        score: detection.confidence,
        x: detection.x,
        y: detection.y,
        w: detection.width,
        h: detection.height,
        cx: f64::from(detection.x) + f64::from(detection.width) * 0.5,
        cy: f64::from(detection.y) + f64::from(detection.height) * 0.5,
    }
}

fn vision_target_state(
    selection: &TargetSelection,
    target_detection_index: Option<usize>,
) -> Option<VisionTargetState> {
    let track_id = selection.target_track_id?.0;
    let class_id = selection.target_class_id?;
    let score = selection.target_detection_confidence?;
    let identity_confidence = selection.target_identity_confidence?;
    let x1 = selection.target_box_x?;
    let y1 = selection.target_box_y?;
    let width = selection.target_box_width?;
    let height = selection.target_box_height?;
    let aim_x = selection.target_aim_x?;
    let aim_y = selection.target_aim_y?;
    Some(VisionTargetState {
        target_detection_index,
        track_id,
        class_id,
        cls: class_id,
        score,
        identity_confidence,
        x1,
        y1,
        x2: x1 + width,
        y2: y1 + height,
        box_cx: x1 + width * 0.5,
        box_cy: y1 + height * 0.5,
        cx: x1 + width * 0.5,
        cy: y1 + height * 0.5,
        observed_aim_x: aim_x,
        observed_aim_y: aim_y,
    })
}

fn target_pipeline_state(selection: &TargetSelection, has_sample: bool) -> TargetPipelineState {
    let mut rejection_reasons = Vec::with_capacity(4);
    if selection.rejected_by_confidence > 0 {
        rejection_reasons.push("confidence");
    }
    if selection.rejected_by_class > 0 {
        rejection_reasons.push("class_filter");
    }
    if selection.rejected_by_aspect_ratio > 0 {
        rejection_reasons.push("ratio_check");
    }
    if selection.rejected_by_fov > 0 {
        rejection_reasons.push("selection_fov");
    }
    let (code, stage, message) = if !has_sample {
        ("NO_SAMPLE", "targeting", "等待 Rust targeting 样本")
    } else if selection.target_track_id.is_some() {
        ("TARGET_SELECTED", "selection", "Rust selector 已选择目标")
    } else if selection.candidates == 0 {
        ("NO_CANDIDATES", "targeting", "当前帧没有检测候选")
    } else if selection.rejected_by_class > 0 && selection.inside_fov == 0 {
        (
            "BASIC_CANDIDATE_REJECTED",
            "class_filter",
            "候选类别未通过 Rust allowlist",
        )
    } else {
        (
            "ASSOCIATION_CANDIDATE_REJECTED",
            "targeting",
            "候选未通过 Rust targeting 准入门",
        )
    };
    TargetPipelineState {
        code,
        stage,
        message,
        rejection_reasons,
        counts: TargetPipelineCounts {
            raw_candidates: selection.candidates,
            eligible_candidates: selection.inside_fov,
            selected_targets: usize::from(selection.target_track_id.is_some()),
        },
    }
}

struct OutputTraceProjection<'a> {
    snapshot: &'a RuntimeSnapshot,
    runtime_config: Option<&'a AppConfig>,
    target_pipeline: &'a TargetPipelineState,
    hardware_output_enabled: bool,
    control_reason: Option<&'static str>,
}

impl OutputTraceProjection<'_> {
    fn state(self) -> OutputTraceState {
        let pipeline = &self.snapshot.pipeline_metrics;
        let running = self.snapshot.pipeline.state == PipelineState::Running;
        let inference_running =
            running && self.snapshot.subsystems.inference.state == SubsystemState::Running;

        if !running {
            return OutputTraceState {
                code: "runtime_stopped",
                state: "idle",
                detail: "主链未运行，尚未产生检测、目标选择或控制输出。",
                next_action: "start_mainline",
            };
        }
        if !inference_running {
            return OutputTraceState {
                code: "inference_not_running",
                state: "blocked",
                detail: "推理子系统未进入运行态，无法产生 DetectionBatch。",
                next_action: "check_model",
            };
        }
        if self.snapshot.perception_metrics.published_batches == 0 {
            return OutputTraceState {
                code: "no_detection_batches",
                state: "waiting",
                detail: "DeepStream 尚未发布 DetectionBatch。",
                next_action: "check_capture_or_model",
            };
        }
        if pipeline.received_batches == 0 && pipeline.targeting_batches == 0 {
            return OutputTraceState {
                code: "runtime_not_consuming_batches",
                state: "waiting",
                detail: "检测批次已发布，但 Rust runtime 尚未消费。",
                next_action: "inspect_runtime_ingress",
            };
        }
        if self
            .snapshot
            .telemetry
            .detection_data_age_ms
            .zip(
                self.runtime_config
                    .map(|config| config.pipeline.freshness_threshold_ms),
            )
            .is_some_and(|(age, threshold)| age > threshold)
        {
            return OutputTraceState {
                code: "stale_detection_batch",
                state: "blocked",
                detail: "最新检测批次超过控制新鲜度阈值。",
                next_action: "check_latency",
            };
        }
        if self.target_pipeline.code != "TARGET_SELECTED" {
            return OutputTraceState {
                code: "target_not_selected",
                state: "blocked",
                detail: "目标选择没有产出可控目标。",
                next_action: "check_targeting",
            };
        }
        if !pipeline.control.sample_available {
            return OutputTraceState {
                code: "control_not_calculated",
                state: "waiting",
                detail: "目标已经选择，但控制器尚未产生命令样本。",
                next_action: "inspect_control",
            };
        }
        if !self.hardware_output_enabled {
            return OutputTraceState {
                code: "hardware_output_disabled",
                state: "blocked",
                detail: "当前构建未启用硬件输出能力。",
                next_action: "check_license_or_build",
            };
        }
        if !pipeline.output_gate_open {
            return OutputTraceState {
                code: "output_gate_closed",
                state: "blocked",
                detail: "物理输出门关闭，控制量不会发送到设备。",
                next_action: "enable_output_gate",
            };
        }
        if !pipeline.control.trigger_active {
            return OutputTraceState {
                code: "trigger_inactive",
                state: "waiting",
                detail: "触发条件未激活，控制器不会发送物理命令。",
                next_action: "activate_trigger",
            };
        }
        if !pipeline.device_connected {
            return OutputTraceState {
                code: "device_not_connected",
                state: "blocked",
                detail: "kmNet 设备未连接，无法发送物理命令。",
                next_action: "connect_kmnet",
            };
        }
        if !pipeline.control.emit_allowed {
            return OutputTraceState {
                code: self
                    .control_reason
                    .filter(|reason| !reason.is_empty())
                    .unwrap_or("control_blocked"),
                state: "blocked",
                detail: "控制器已计算样本，但当前样本不允许发送。",
                next_action: "inspect_control",
            };
        }
        match pipeline.output_delivery_state {
            OutputDeliveryState::GenerationFenced => OutputTraceState {
                code: "generation_fenced",
                state: "waiting",
                detail: "配置刚更新，旧 generation 命令已丢弃，等待下一帧。",
                next_action: "wait_next_frame",
            },
            OutputDeliveryState::SendFailed => OutputTraceState {
                code: "device_send_failed",
                state: "blocked",
                detail: "最近一次设备发送失败，等待设备恢复。",
                next_action: "connect_kmnet",
            },
            OutputDeliveryState::DeviceDisabled => OutputTraceState {
                code: "device_output_disabled",
                state: "blocked",
                detail: "设备输出已停用。",
                next_action: "connect_kmnet",
            },
            OutputDeliveryState::Idle
            | OutputDeliveryState::GateClosed
            | OutputDeliveryState::TriggerInactive
            | OutputDeliveryState::NoMovement
            | OutputDeliveryState::Superseded
            | OutputDeliveryState::Sent => OutputTraceState {
                code: "ready",
                state: "ready",
                detail: "检测、目标选择、控制器、输出门和设备连接均已贯通。",
                next_action: "monitor_output",
            },
        }
    }
}

const fn lock_reason_label(reason: LockReason) -> &'static str {
    match reason {
        LockReason::PreferredClass => "PREFERRED_CLASS",
        LockReason::FallbackClass => "FALLBACK_CLASS",
    }
}

const fn control_mode_label(mode: ControlMode) -> &'static str {
    match mode {
        ControlMode::Continuous => "CONTINUOUS",
    }
}

const fn block_reason_label(reason: BlockReason) -> &'static str {
    match reason {
        BlockReason::TimestampDomainInvalid => "TIMESTAMP_DOMAIN_INVALID",
        BlockReason::StaleObservation => "STALE_OBSERVATION",
        BlockReason::NonMonotonicObservation => "NON_MONOTONIC_OBSERVATION",
        BlockReason::TargetInvalid => "TARGET_INVALID",
        BlockReason::GeometryInvalid => "GEOMETRY_INVALID",
        BlockReason::TriggerInactive => "TRIGGER_INACTIVE",
        BlockReason::TriggerDelayPending => "TRIGGER_DELAY_PENDING",
        BlockReason::DeadZone => "DEAD_ZONE",
        BlockReason::DemandOutOfRange => "DEMAND_OUT_OF_RANGE",
        BlockReason::None => "",
    }
}

fn subsystem_can_run(subsystem: &SubsystemSnapshot) -> bool {
    !matches!(
        subsystem.state,
        SubsystemState::Failed | SubsystemState::Unavailable
    )
}

fn first_subsystem_error(
    snapshot: &RuntimeSnapshot,
    hardware_output_enabled: bool,
) -> Option<RuntimeErrorSummary> {
    [
        &snapshot.subsystems.capture,
        &snapshot.subsystems.inference,
        &snapshot.subsystems.control,
        &snapshot.subsystems.device,
    ]
    .into_iter()
    .find_map(|subsystem| {
        matches!(
            subsystem.state,
            SubsystemState::Failed | SubsystemState::Unavailable
        )
        .then(|| subsystem.last_error.clone())
        .flatten()
        .filter(|error| {
            error.code != "device_uncommissioned"
                && (error.code != "perception_adapter_unavailable" || hardware_output_enabled)
        })
    })
}

fn serialized_label(value: &impl Serialize) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned())
}

#[cfg(test)]
mod tests {
    use novasight_core::Generation;
    use novasight_core::controller::recoil::{RecoilBlockReason, RecoilDecision, RecoilState};
    use novasight_core::controller::{AimResult, BlockReason, ControlMode};
    use novasight_core::prediction::PredictionMotionState;
    use novasight_core::tracking::{LockReason, TargetSelection, TrackId};
    use novasight_pipeline::DetectionTelemetryItem;
    use novasight_runtime::{
        AppConfig, PipelineState, RuntimeErrorSummary, RuntimeSnapshot, SubsystemState,
    };

    use super::RuntimeStatusState;

    #[test]
    fn degraded_kmnet_is_disconnected_retryable_and_exposes_the_runtime_error() {
        let mut snapshot = RuntimeSnapshot::default();
        snapshot.pipeline.state = PipelineState::Running;
        snapshot.subsystems.device.state = SubsystemState::Degraded;
        snapshot.subsystems.device.last_error = Some(RuntimeErrorSummary::new(
            "device_reconnecting",
            "kmNet helper timed out",
        ));
        let config = AppConfig {
            device: Some(Default::default()),
            ..AppConfig::default()
        };

        let value = serde_json::to_value(RuntimeStatusState::new(
            &snapshot,
            Some(&config),
            Some(0),
            true,
            None,
            None,
        ))
        .unwrap();
        let kmnet = &value["executor"]["executors"]["kmnet"];
        assert_eq!(kmnet["available"], true);
        assert_eq!(kmnet["connected"], false);
        assert_eq!(kmnet["buttons_available"], false);
        assert_eq!(kmnet["connection_state"], "degraded");
        assert_eq!(kmnet["retryable"], true);
        assert_eq!(kmnet["last_error"], "kmNet helper timed out");
        assert!(value["fatal_error"].is_null());
    }

    #[test]
    fn absent_perception_adapter_is_an_explicit_non_fatal_runtime_mode() {
        let mut snapshot = RuntimeSnapshot::default();
        snapshot.pipeline.state = PipelineState::Running;
        snapshot.subsystems.capture.state = SubsystemState::Unavailable;
        snapshot.subsystems.inference.state = SubsystemState::Unavailable;
        snapshot.subsystems.inference.last_error = Some(RuntimeErrorSummary::new(
            "perception_adapter_unavailable",
            "no perception adapter is installed for this daemon mode",
        ));
        let config = AppConfig {
            inference: Some(Default::default()),
            ..AppConfig::default()
        };

        let value = serde_json::to_value(RuntimeStatusState::new(
            &snapshot,
            Some(&config),
            Some(0),
            false,
            None,
            None,
        ))
        .unwrap();

        assert_eq!(value["semantic"]["phase"], "running");
        assert_eq!(value["semantic"]["perception_phase"], "unavailable");
        assert_eq!(value["inference"]["available"], false);
        assert!(value["fatal_error"].is_null());

        let hardware_value = serde_json::to_value(RuntimeStatusState::new(
            &snapshot,
            Some(&config),
            Some(0),
            true,
            None,
            None,
        ))
        .unwrap();
        assert_eq!(
            hardware_value["fatal_error"]["code"],
            "perception_adapter_unavailable"
        );
    }

    #[test]
    fn projects_daemon_owned_control_telemetry_into_studio_shape() {
        let mut snapshot = RuntimeSnapshot::default();
        snapshot.pipeline_metrics.control = AimResult {
            sample_available: true,
            aim_x: 330.0,
            aim_y: 317.0,
            observation_width: 640,
            observation_height: 640,
            trigger_active: true,
            dx: 12,
            dy: -3,
            emit_allowed: true,
            block_reason: BlockReason::None,
            mode: ControlMode::Continuous,
            velocity_x: 0.25,
            history_position_count: 4,
            velocity_samples: [Some(0.2), Some(0.3), Some(0.25)],
            medoid_velocity: Some(0.25),
            motion_state: PredictionMotionState::Continuous,
            measurement_dt_ms: Some(8.0),
            reference_dt_ms: 8.1,
            prediction_lead_ms: 1.0,
            prediction_raw_offset_x: 2.0,
            prediction_allowed_cap_x: 3.0,
            prediction_allowed: true,
            velocity_y: -0.10,
            velocity_samples_y: [Some(-0.08), Some(-0.12), Some(-0.10)],
            medoid_velocity_y: Some(-0.10),
            motion_state_y: PredictionMotionState::Unstable,
            measurement_dt_ms_y: Some(8.0),
            reference_dt_ms_y: 8.1,
            prediction_raw_offset_y: -0.8,
            prediction_allowed_cap_y: 1.5,
            prediction_allowed_y: true,
            predicted_offset_x: 1.6,
            predicted_offset_y: -0.6,
            observed_error_x: 10.0,
            observed_error_y: -3.0,
            filtered_error_x: 11.6,
            filtered_error_y: -3.0,
            full_error_counts_x: 28.0,
            full_error_counts_y: -7.0,
            float_demand_x: 12.4,
            float_demand_y: -2.9,
            quantizer_residual_x: 0.4,
            quantizer_residual_y: -0.1,
            ..AimResult::default()
        };
        snapshot.pipeline_metrics.recoil = RecoilDecision {
            state: RecoilState::Applied,
            interval_ms: 16,
            configured_y_counts: 2,
            elapsed_since_output_ms: Some(17.0),
            remaining_ms: 0.0,
            requested_counts_y: 2,
            emitted_counts_y: 2,
            source_generation: Some(9),
            block_reason: RecoilBlockReason::None,
        };
        snapshot.pipeline_metrics.targeting_batches = 1;
        snapshot.pipeline.state = PipelineState::Running;
        snapshot.subsystems.inference.state = SubsystemState::Running;
        snapshot.perception_metrics.published_batches = 1;
        snapshot.pipeline_metrics.received_batches = 1;
        snapshot.pipeline_metrics.output_gate_open = true;
        snapshot.pipeline_metrics.device_connected = true;
        snapshot.pipeline_metrics.detections.generation = Some(Generation(9));
        snapshot.pipeline_metrics.detections.coordinate_width = 960;
        snapshot.pipeline_metrics.detections.coordinate_height = 544;
        snapshot.pipeline_metrics.detections.items = vec![DetectionTelemetryItem {
            object_id: 91,
            class_id: 2,
            x: 300.0,
            y: 200.0,
            width: 60.0,
            height: 120.0,
            confidence: 0.91,
        }];
        snapshot.pipeline_metrics.target_selection = TargetSelection {
            target_object_id: Some(91),
            target_track_id: Some(TrackId(17)),
            target_class_id: Some(2),
            target_detection_confidence: Some(0.91),
            target_identity_confidence: Some(0.82),
            target_aim_x: Some(330.0),
            target_aim_y: Some(240.0),
            target_box_x: Some(300.0),
            target_box_y: Some(200.0),
            target_box_width: Some(60.0),
            target_box_height: Some(120.0),
            lock_reason: Some(LockReason::FallbackClass),
            candidates: 3,
            inside_fov: 1,
            rejected_class_ids: vec![4],
            rejected_by_class: 1,
            rejected_by_fov: 1,
            ..TargetSelection::default()
        };

        let value = serde_json::to_value(RuntimeStatusState::new(
            &snapshot, None, None, true, None, None,
        ))
        .unwrap();
        let pipeline = &value["vision"]["control"]["pipeline"];
        let control = &value["vision"]["control"];
        let target = &value["vision"]["target"];
        let target_pipeline = &value["vision"]["target_pipeline"];
        assert_eq!(control["will_emit"], true);
        assert_eq!(control["aim_x"], 331.6);
        assert_eq!(control["mouse_observation"]["measurement_dt_s"], 0.008);
        assert_eq!(control["selector_state"], "LOCKED");
        assert_eq!(control["selection_reason"], "FALLBACK_CLASS");
        assert_eq!(
            control["candidate_filter"]["basic"]["rejected_class_ids"][0],
            4
        );
        assert_eq!(target["track_id"], 17);
        assert_eq!(target["target_detection_index"], 0);
        assert_eq!(target["class_id"], 2);
        assert_eq!(target["x2"], 360.0);
        assert_eq!(target["observed_aim_y"], 240.0);
        assert_eq!(target_pipeline["code"], "TARGET_SELECTED");
        assert_eq!(target_pipeline["counts"]["eligible_candidates"], 1);
        assert_eq!(value["vision"]["output_trace"]["code"], "ready");
        assert_eq!(
            value["vision"]["output_trace"]["next_action"],
            "monitor_output"
        );
        assert_eq!(value["vision"]["inference"]["input_width"], 960);
        assert_eq!(value["vision"]["inference"]["input_height"], 544);
        assert_eq!(value["vision"]["inference"]["generation"], 9);
        assert_eq!(value["vision"]["detection_items"][0]["object_id"], 91);
        assert_eq!(value["vision"]["detection_items"][0]["cx"], 330.0);
        assert_eq!(pipeline["control_mode"], "continuous_atan_medoid_v2");
        assert_eq!(pipeline["mode"], "CONTINUOUS");
        assert_eq!(pipeline["velocity_2"], 0.3);
        assert_eq!(pipeline["motion_state"], "continuous");
        assert_eq!(pipeline["prediction_safe_offset_x"], 1.6);
        assert_eq!(pipeline["prediction_velocity_y"], -0.10);
        assert_eq!(pipeline["motion_state_y"], "unstable");
        assert_eq!(pipeline["prediction_safe_offset_y"], -0.6);
        assert_eq!(pipeline["prediction_allowed_y"], true);
        assert_eq!(pipeline["integer_command_x"], 12);
        assert_eq!(pipeline["quantizer_residual_y"], -0.1);
        assert_eq!(pipeline["block_reason"], "");
        assert_eq!(pipeline["recoil_state"], "APPLIED");
        assert_eq!(pipeline["recoil_active"], true);
        assert_eq!(pipeline["recoil_interval_ms"], 16);
        assert_eq!(pipeline["recoil_y_counts"], 2);
        assert_eq!(pipeline["recoil_emitted_counts_y"], 2);
        assert_eq!(pipeline["recoil_source_generation"], 9);
        assert_eq!(pipeline["recoil_block_reason"], "");
    }

    #[test]
    fn projects_rust_class_rejections_without_inventing_a_target() {
        let mut snapshot = RuntimeSnapshot::default();
        snapshot.pipeline.state = PipelineState::Running;
        snapshot.subsystems.inference.state = SubsystemState::Running;
        snapshot.perception_metrics.published_batches = 1;
        snapshot.pipeline_metrics.received_batches = 1;
        snapshot.pipeline_metrics.targeting_batches = 1;
        snapshot.pipeline_metrics.target_selection = TargetSelection {
            candidates: 3,
            rejected_class_ids: vec![0, 2],
            rejected_by_class: 2,
            rejected_by_confidence: 1,
            ..TargetSelection::default()
        };
        let mut config = AppConfig::default();
        config.pipeline.target_class_filter = "1".to_owned();

        let value = serde_json::to_value(RuntimeStatusState::new(
            &snapshot,
            Some(&config),
            Some(0),
            false,
            None,
            None,
        ))
        .unwrap();

        assert!(value["vision"]["target"].is_null());
        assert_eq!(
            value["vision"]["target_pipeline"]["code"],
            "BASIC_CANDIDATE_REJECTED"
        );
        assert_eq!(
            value["vision"]["target_pipeline"]["rejection_reasons"],
            serde_json::json!(["confidence", "class_filter"])
        );
        assert_eq!(
            value["vision"]["output_trace"]["code"],
            "target_not_selected"
        );
        assert_eq!(
            value["vision"]["output_trace"]["next_action"],
            "check_targeting"
        );
        assert_eq!(
            value["vision"]["control"]["candidate_filter"]["effective_class_filter"],
            "1"
        );
        assert_eq!(
            value["vision"]["control"]["candidate_filter"]["basic"]["rejected_class_ids"],
            serde_json::json!([0, 2])
        );
    }
}
