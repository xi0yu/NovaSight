use novasight_core::{ErrorSnapshot, OperationalSnapshot, RunIntent, RuntimeCommandReceipt};
use serde::Serialize;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ErrorResponse {
    pub code: String,
    pub message: String,
}

impl ErrorResponse {
    pub(crate) fn new(code: &'static str, message: String) -> Self {
        Self {
            code: code.to_owned(),
            message,
        }
    }
}

impl From<&ErrorSnapshot> for ErrorResponse {
    fn from(error: &ErrorSnapshot) -> Self {
        Self {
            code: error.code.clone(),
            message: error.message.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExecutorAvailabilityResponse {
    pub available: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExecutorCollectionResponse {
    pub dry_run: ExecutorAvailabilityResponse,
    pub kmnet: ExecutorAvailabilityResponse,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExecutorStatusResponse {
    pub selected: String,
    pub executors: ExecutorCollectionResponse,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CaptureStateResponse {
    pub available: bool,
    pub device: String,
    pub profile: Option<CaptureProfileResponse>,
    pub backend: Option<String>,
    pub fps_capture: f64,
    pub frame_period_ms: f64,
    pub capture_wait_ms: f64,
    pub frames_dropped: u64,
    pub preview_target_fps: f64,
    pub preview_fps: f64,
    pub preview_frames: u64,
    pub preview_output_frames: u64,
    pub preview_dropped: u64,
    pub recoveries: u64,
    pub last_error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CaptureProfileResponse {
    pub pixel_format: String,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub preference: String,
    pub selection_reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StatisticsResponse {
    pub capture_counter: u64,
    pub inference_counter: u64,
    pub detection_batch_counter: u64,
    pub detection_batch_consumed_counter: u64,
    pub dropped_counter: u64,
    pub skipped_counter: u64,
    pub capture_fps: f64,
    pub inference_fps: f64,
    pub e2e_latency: f64,
    pub control_observation_counter: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct InferenceStateResponse {
    pub available: bool,
    pub configured: bool,
    pub running: bool,
    pub mode: String,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeConfigSummaryResponse {
    pub version: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PipelineStateResponse {
    pub running: bool,
    pub phase: novasight_core::RuntimePhase,
    pub run_intent: bool,
    pub epoch: Option<u64>,
    pub source: String,
    pub last_generation: Option<u64>,
    pub processed_batches: u64,
    pub device_receipts: u64,
    pub mode: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RuntimePowerSavingResponse {
    pub enabled: bool,
    pub mode: String,
    pub run_intent: bool,
    pub suspended_by_policy: bool,
    pub running: bool,
    pub host_id: String,
    pub target_host_id: String,
    pub host_online: bool,
    pub heartbeat_age_ms: Option<f64>,
    pub auto_resume: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VisionStateResponse {
    pub available: bool,
    pub mode: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RuntimeStateResponse {
    pub running: bool,
    pub source: String,
    pub active_model: Option<serde_json::Value>,
    pub executor: ExecutorStatusResponse,
    pub capture: CaptureStateResponse,
    pub statistics: StatisticsResponse,
    pub inference: InferenceStateResponse,
    pub config: RuntimeConfigSummaryResponse,
    pub pipeline: PipelineStateResponse,
    pub power_saving: RuntimePowerSavingResponse,
    pub vision: VisionStateResponse,
    pub fatal_error: Option<ErrorResponse>,
}

impl From<&OperationalSnapshot> for RuntimeStateResponse {
    fn from(snapshot: &OperationalSnapshot) -> Self {
        let run_intent = snapshot.run_intent == RunIntent::Running;
        Self {
            running: snapshot.running,
            source: snapshot.source.clone(),
            active_model: None,
            executor: ExecutorStatusResponse {
                selected: "dry_run".to_owned(),
                executors: ExecutorCollectionResponse {
                    dry_run: ExecutorAvailabilityResponse { available: true },
                    kmnet: ExecutorAvailabilityResponse { available: false },
                },
            },
            capture: CaptureStateResponse {
                available: false,
                device: "replay".to_owned(),
                profile: None,
                backend: Some("replay".to_owned()),
                fps_capture: 0.0,
                frame_period_ms: 0.0,
                capture_wait_ms: 0.0,
                frames_dropped: 0,
                preview_target_fps: 0.0,
                preview_fps: 0.0,
                preview_frames: 0,
                preview_output_frames: 0,
                preview_dropped: 0,
                recoveries: 0,
                last_error: None,
            },
            statistics: StatisticsResponse {
                capture_counter: 0,
                inference_counter: snapshot.processed_batches,
                detection_batch_counter: snapshot.processed_batches,
                detection_batch_consumed_counter: snapshot.processed_batches,
                dropped_counter: 0,
                skipped_counter: 0,
                capture_fps: 0.0,
                inference_fps: 0.0,
                e2e_latency: 0.0,
                control_observation_counter: snapshot.device_receipts,
            },
            inference: InferenceStateResponse {
                available: false,
                configured: false,
                running: snapshot.running,
                mode: "replay".to_owned(),
                reason: "TensorRT is unavailable in Phase 1 replay mode".to_owned(),
            },
            config: RuntimeConfigSummaryResponse { version: 1 },
            pipeline: PipelineStateResponse {
                running: snapshot.running,
                phase: snapshot.phase,
                run_intent,
                epoch: snapshot.epoch.map(|epoch| epoch.0),
                source: snapshot.source.clone(),
                last_generation: snapshot.last_generation.map(|generation| generation.0),
                processed_batches: snapshot.processed_batches,
                device_receipts: snapshot.device_receipts,
                mode: "replay".to_owned(),
            },
            power_saving: RuntimePowerSavingResponse {
                enabled: false,
                mode: "disabled".to_owned(),
                run_intent,
                suspended_by_policy: false,
                running: snapshot.running,
                host_id: String::new(),
                target_host_id: String::new(),
                host_online: false,
                heartbeat_age_ms: None,
                auto_resume: false,
                reason: "unavailable in Phase 1 replay mode".to_owned(),
            },
            vision: VisionStateResponse {
                available: false,
                mode: "replay".to_owned(),
                reason: "online vision hardware is unavailable in Phase 1 replay mode".to_owned(),
            },
            fatal_error: snapshot.fatal_error.as_ref().map(ErrorResponse::from),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeStartResponse {
    pub running: bool,
    pub accepted: bool,
    pub failed: bool,
    pub epoch: Option<u64>,
    pub operation_id: Option<String>,
}

impl RuntimeStartResponse {
    pub(crate) fn accepted(receipt: RuntimeCommandReceipt, running: bool) -> Self {
        Self {
            running,
            accepted: true,
            failed: false,
            epoch: receipt.epoch.map(|epoch| epoch.0),
            operation_id: None,
        }
    }
}
