//! Immutable daemon-owned runtime snapshot, including the counters needed to
//! explain freshness loss and output suppression on a deployed system.

use serde::{Deserialize, Serialize};

use novasight_core::RuntimeEpoch;
use novasight_pipeline::{PerceptionMetrics, PipelineMetrics};
use novasight_store::model_catalog::ActiveModelDeployment;

use crate::protocol::RuntimeErrorSummary;
use crate::state::{DaemonState, PipelineState};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonSnapshot {
    pub state: DaemonState,
    pub version: String,
    pub uptime_ms: u64,
}

impl Default for DaemonSnapshot {
    fn default() -> Self {
        Self {
            state: DaemonState::default(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            uptime_ms: 0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineSnapshot {
    pub state: PipelineState,
    pub epoch: Option<RuntimeEpoch>,
    pub started_at_ms: Option<u64>,
    pub last_error: Option<RuntimeErrorSummary>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubsystemSnapshot {
    pub state: crate::protocol::SubsystemState,
    pub last_error: Option<RuntimeErrorSummary>,
    pub restart_count: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubsystemSnapshots {
    pub capture: SubsystemSnapshot,
    pub inference: SubsystemSnapshot,
    pub control: SubsystemSnapshot,
    pub device: SubsystemSnapshot,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    /// Strictly monotonic process-local publication order. Consumers use this
    /// instead of wall time when rejecting stale REST/WebSocket frames.
    #[serde(default)]
    pub sequence: u64,
    pub daemon: DaemonSnapshot,
    pub pipeline: PipelineSnapshot,
    pub subsystems: SubsystemSnapshots,
    #[serde(default)]
    pub perception_metrics: PerceptionMetrics,
    #[serde(default)]
    pub pipeline_metrics: PipelineMetrics,
    #[serde(default)]
    pub device_metrics: DeviceMetrics,
    #[serde(default)]
    pub model: ModelSnapshot,
    /// Human-facing rates and freshness derived on the supervisor control
    /// plane. Producers never calculate or publish these values per frame.
    #[serde(default)]
    pub telemetry: RuntimeTelemetrySnapshot,
    pub updated_at_ms: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RuntimeTelemetrySnapshot {
    pub sample_window_ms: Option<u64>,
    pub nvinfer_input_fps: Option<f64>,
    #[serde(default)]
    pub nvinfer_output_fps: Option<f64>,
    pub detection_batch_fps: Option<f64>,
    /// Detection batches entering target selection per second. This is not a
    /// count of emitted control decisions or device commands.
    pub targeting_batch_fps: Option<f64>,
    pub detection_data_age_ms: Option<f64>,
    /// Latest correlated nvinfer sink-to-src duration. This includes the
    /// element's preprocessing, TensorRT execution, and parser work.
    pub inference_latency_ms: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMetrics {
    pub diagnostic_move_count: u64,
    pub last_diagnostic_dx: Option<i32>,
    pub last_diagnostic_dy: Option<i32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSnapshot {
    pub active: Option<ActiveModelDeployment>,
    pub catalog_error: Option<String>,
    pub input_width: Option<u32>,
    pub input_height: Option<u32>,
}
