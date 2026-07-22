//! Runtime snapshot. Distinct DTOs per the proposal: the
//! RuntimeSnapshot bundles daemon, pipeline, and subsystem state
//! without KPI fields; FPS, latency, and dropped-frame counters
//! live in a separate MetricsSnapshot (added in a later commit).

use serde::{Deserialize, Serialize};

use novasight_core::RuntimeEpoch;
use novasight_pipeline::PerceptionMetrics;

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

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub daemon: DaemonSnapshot,
    pub pipeline: PipelineSnapshot,
    pub subsystems: SubsystemSnapshots,
    #[serde(default)]
    pub perception_metrics: PerceptionMetrics,
    pub updated_at_ms: u64,
}
