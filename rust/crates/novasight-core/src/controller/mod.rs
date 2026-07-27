//! Production target-to-command controller.
//!
//! Prediction supplies one selected target position. This module converts its
//! two-axis error into a bounded continuous demand, then composes the limiter
//! that makes it representable as integer counts. Hardware delivery and stale
//! command replacement remain downstream concerns.

mod atan;
pub mod recoil;
pub mod replay;
mod response_curve;

pub use atan::{
    ActuationFeedback, BlockReason, ControlDecision, ControlMode, ControlObservation,
    DualPhaseConfig, DualPhaseControl,
};
