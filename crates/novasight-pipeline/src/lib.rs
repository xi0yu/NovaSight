//! `novasight-pipeline` — capture / inference / control / device
//! orchestration on dedicated `std::thread` workers, with typed
//! `LatestSlot<T>` data flow between stages.
//!
//! The public ingress and runtime interfaces hide the worker topology.
//! Hot-path stage handoff uses capacity-one latest slots so a slow
//! consumer never forces capture to queue stale observations.

#![forbid(unsafe_code)]

mod crosshair;
mod perception;
mod preview;
mod runtime;
mod slot;

pub use crosshair::{
    ControlReference, CrosshairConfig, CrosshairEpoch, CrosshairError, CrosshairFramePublisher,
    CrosshairHub, CrosshairObservation, CrosshairSnapshot, CrosshairTemplateSummary,
};
pub use perception::{
    ModelCandidate, ParserContract, PerceptionAdapter, PerceptionError, PerceptionErrorKind,
    PerceptionEvent, PerceptionMetrics, PerceptionModelContract, PerceptionRuntimeContract,
    PerceptionSession, validate_parser_preset,
};
pub use preview::{PreviewError, PreviewFrame, PreviewHub, PreviewSnapshot, PreviewSubscription};

pub use runtime::{
    DetectionTelemetry, DetectionTelemetryItem, PipelineConfig, PipelineError, PipelineEvent,
    PipelineIngress, PipelineLiveConfig, PipelineMetrics, PipelineRuntime, PipelineStatus,
    TriggerMode,
};
pub use slot::{LatestSlot, SlotClosed, SlotMetrics, TryPublishError};
