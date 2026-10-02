//! Daemon and pipeline state enums. Distinct from the snapshot
//! families so the wire DTOs can carry every transition without
//! aliasing the runtime-internal enums.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonState {
    #[default]
    Starting,
    Ready,
    ShuttingDown,
    Failed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineState {
    #[default]
    Stopped,
    Starting,
    Running,
    Standby,
    Stopping,
    Faulted,
}
