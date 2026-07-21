use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{ErrorSnapshot, Generation, RuntimeEpoch};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimePhase {
    Stopped,
    Starting,
    Running,
    Standby,
    Stopping,
    Faulted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunIntent {
    Stopped,
    Running,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OperationalSnapshot {
    pub phase: RuntimePhase,
    pub run_intent: RunIntent,
    pub epoch: Option<RuntimeEpoch>,
    pub running: bool,
    pub source: String,
    pub last_generation: Option<Generation>,
    pub processed_batches: u64,
    pub device_receipts: u64,
    pub fatal_error: Option<ErrorSnapshot>,
}

impl OperationalSnapshot {
    pub(crate) fn initial_replay() -> Self {
        Self {
            phase: RuntimePhase::Stopped,
            run_intent: RunIntent::Stopped,
            epoch: None,
            running: false,
            source: "replay".to_owned(),
            last_generation: None,
            processed_batches: 0,
            device_receipts: 0,
            fatal_error: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RuntimeCommandReceipt {
    pub epoch: Option<RuntimeEpoch>,
    pub phase: RuntimePhase,
    pub snapshot: Arc<OperationalSnapshot>,
}
