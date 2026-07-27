//! Production target-to-command controller.
//!
//! Prediction supplies one selected target position. This module converts its
//! two-axis error into a bounded continuous demand and integer device counts.
//! Device rate policy and hardware delivery remain downstream concerns.

mod atan;
pub mod recoil;
pub mod replay;

pub use atan::{
    ActuationFeedback, BlockReason, ControlDecision, ControlMode, ControlObservation,
    DualPhaseConfig, DualPhaseControl,
};
