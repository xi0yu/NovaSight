//! Wire protocol types shared by the API, the CLI client, and the
//! supervisor. The snapshot bodies live in `snapshot`; the runtime
//! command variants live in `command`. This module carries only the
//! cross-cutting wire types: subsystem state enum, the error
//! summary, and the small DTOs that need to be visible at the API
//! layer and the supervisor boundary.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum SubsystemState {
    #[default]
    Stopped,
    Starting,
    Ready,
    Running,
    Degraded,
    Stopping,
    Failed,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeErrorSummary {
    pub code: String,
    pub message: String,
    pub subsystem: Option<String>,
}

impl RuntimeErrorSummary {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            subsystem: None,
        }
    }
}
