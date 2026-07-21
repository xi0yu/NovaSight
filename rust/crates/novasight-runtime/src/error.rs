//! Runtime error envelope. `RuntimeError` is the typed failure
//! returned by every `RuntimeHandle` method; `RuntimeErrorKind`
//! classifies the error so tests and HTTP handlers can map it to
//! stable exit codes and HTTP status.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::protocol::RuntimeErrorSummary;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RuntimeErrorKind {
    /// The supervisor task is no longer running.
    SupervisorUnavailable,
    /// The supervisor dropped a command before it could be
    /// delivered to the queue.
    SupervisorClosed,
    /// The supervisor's oneshot reply channel was dropped before a
    /// reply landed.
    SupervisorReplyLost,
    /// Pipeline cannot satisfy the requested transition.
    InvalidPipelineState,
    /// No new runtime epoch can be allocated without wrapping.
    RuntimeEpochExhausted,
    /// Generic runtime error. New variants land in later commits.
    Other,
}

impl RuntimeErrorKind {
    pub fn code(self) -> &'static str {
        match self {
            RuntimeErrorKind::SupervisorUnavailable => "supervisor_unavailable",
            RuntimeErrorKind::SupervisorClosed => "supervisor_closed",
            RuntimeErrorKind::SupervisorReplyLost => "supervisor_reply_lost",
            RuntimeErrorKind::InvalidPipelineState => "invalid_pipeline_state",
            RuntimeErrorKind::RuntimeEpochExhausted => "runtime_epoch_exhausted",
            RuntimeErrorKind::Other => "runtime_error",
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq, Serialize, Deserialize)]
#[error("{kind:?}: {message}")]
pub struct RuntimeError {
    pub kind: RuntimeErrorKind,
    pub message: String,
}

impl RuntimeError {
    pub fn new(kind: RuntimeErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn supervisor_unavailable() -> Self {
        Self::new(
            RuntimeErrorKind::SupervisorUnavailable,
            "runtime supervisor task is not running",
        )
    }

    pub fn supervisor_closed() -> Self {
        Self::new(
            RuntimeErrorKind::SupervisorClosed,
            "runtime supervisor dropped the command channel",
        )
    }

    pub fn supervisor_reply_lost() -> Self {
        Self::new(
            RuntimeErrorKind::SupervisorReplyLost,
            "runtime supervisor reply channel closed before a response arrived",
        )
    }

    pub fn invalid_pipeline_state(message: impl Into<String>) -> Self {
        Self::new(RuntimeErrorKind::InvalidPipelineState, message)
    }

    pub fn runtime_epoch_exhausted() -> Self {
        Self::new(
            RuntimeErrorKind::RuntimeEpochExhausted,
            "runtime epoch counter is exhausted",
        )
    }

    pub fn summary(&self) -> RuntimeErrorSummary {
        RuntimeErrorSummary::new(self.kind.code(), self.message.clone())
    }
}
