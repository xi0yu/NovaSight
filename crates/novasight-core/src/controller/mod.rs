//! Production aim algorithm and its pure numeric control law.
//!
//! `AimAlgorithm` is the stateful interface used by runtime adapters.
//! `AimControlLaw` is the stateless formula interface from measured error and
//! predicted displacement to continuous device-count demand. Hardware delivery
//! and stale command replacement remain downstream concerns.

mod algorithm;
mod control_law;
pub mod recoil;
#[cfg(feature = "replay-tools")]
pub mod replay;

pub use algorithm::{
    AimAlgorithm, AimAlgorithmConfig, AimResult, AimSample, BlockReason, ControlMode,
};
pub use control_law::{
    AimControlInput, AimControlLaw, AimControlParameters, AimControlResult, AxisPair,
    DEFAULT_ATAN_SCALE_COUNTS,
};
