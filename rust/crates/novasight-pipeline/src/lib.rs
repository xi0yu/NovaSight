//! `novasight-pipeline` — capture / inference / control / device
//! orchestration on dedicated `std::thread` workers, with typed
//! `LatestSlot<T>` data flow between stages.
//!
//! The public ingress and runtime interfaces hide the worker topology.
//! Hot-path stage handoff uses capacity-one latest slots so a slow
//! consumer never forces capture to queue stale observations.

#![forbid(unsafe_code)]

mod runtime;
mod slot;

pub use runtime::{
    PipelineConfig, PipelineError, PipelineIngress, PipelineMetrics, PipelineRuntime,
    PipelineStatus,
};
pub use slot::{LatestSlot, SlotClosed, SlotMetrics};
