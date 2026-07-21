use serde::{Deserialize, Serialize};

use crate::AppError;

/// Immutable, transport-safe description of a terminal runtime failure.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ErrorSnapshot {
    pub code: String,
    pub message: String,
}

impl ErrorSnapshot {
    pub(crate) fn from_error(error: &AppError) -> Self {
        let code = match error {
            AppError::RuntimeCommandConflict { .. } => "runtime_command_conflict",
            AppError::RuntimeEpochMismatch { .. } => "runtime_epoch_mismatch",
            AppError::RuntimeManagerUnavailable => "runtime_manager_unavailable",
            AppError::RuntimeTaskTerminated => "runtime_task_terminated",
            _ => "runtime_session_fault",
        };
        Self {
            code: code.to_owned(),
            message: error.to_string(),
        }
    }
}
