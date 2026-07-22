#![deny(unsafe_op_in_unsafe_fn)]

mod clock;
pub mod deepstream;
pub mod kmnet;
pub mod kmnet_native;

pub use clock::SystemMonotonicClock;
