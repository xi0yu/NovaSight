//! `novasight-pipeline` — capture / inference / control / device
//! orchestration on dedicated `std::thread` workers, with typed
//! `LatestSlot<T>` data flow between stages.
//!
//! The public ingress and runtime interfaces hide the worker topology.
//! Hot-path stage handoff uses capacity-one latest slots so a slow
//! consumer never forces capture to queue stale observations.

#![forbid(unsafe_code)]

mod perception;
mod runtime;
mod slot;

pub use perception::{
    PerceptionAdapter, PerceptionError, PerceptionEvent, PerceptionMetrics, PerceptionSession,
};

pub use runtime::{
    PipelineConfig, PipelineError, PipelineEvent, PipelineIngress, PipelineMetrics,
    PipelineRuntime, PipelineStatus,
};
pub use slot::{LatestSlot, SlotClosed, SlotMetrics, TryPublishError};
