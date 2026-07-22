#![deny(unsafe_op_in_unsafe_fn)]

mod clock;
pub mod deepstream;

pub use clock::SystemMonotonicClock;
