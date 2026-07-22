use std::sync::Arc;
use std::sync::mpsc::SyncSender;

use novasight_core::{Clock, RuntimeEpoch};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::PipelineIngress;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PerceptionEvent {
    Faulted { message: String },
    Stopped,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PerceptionMetrics {
    pub probed_buffers: u64,
    pub published_batches: u64,
    pub busy_dropped_batches: u64,
    pub overwritten_snapshots: u64,
    pub unavailable_snapshot_slots: u64,
    pub extraction_rejections: u64,
    pub admission_rejections: u64,
    pub ingress_rejections: u64,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("perception adapter failed: {message}")]
pub struct PerceptionError {
    message: String,
}

impl PerceptionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Factory for one epoch-scoped perception producer.
pub trait PerceptionAdapter: Send + Sync + 'static {
    fn start(
        &self,
        epoch: RuntimeEpoch,
        ingress: PipelineIngress,
        clock: Arc<dyn Clock>,
        events: SyncSender<PerceptionEvent>,
    ) -> Result<Box<dyn PerceptionSession>, PerceptionError>;
}

/// Running perception resources owned by the RuntimeSupervisor.
pub trait PerceptionSession: Send + 'static {
    fn metrics(&self) -> PerceptionMetrics {
        PerceptionMetrics::default()
    }

    /// Stop producing before the post-inference pipeline is closed.
    fn shutdown(&mut self) -> Result<(), PerceptionError>;
}
