//! Daemon and pipeline state enums. Distinct from the snapshot
//! families so the wire DTOs can carry every transition without
//! aliasing the runtime-internal enums.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum DaemonState {
    #[default]
    Starting,
    Ready,
    ShuttingDown,
    Failed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum PipelineState {
    #[default]
    Stopped,
    Starting,
    Running,
    Standby,
    Stopping,
    Faulted,
}
