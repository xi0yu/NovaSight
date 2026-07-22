//! NovaSight's unique production composition root.

mod application;
mod server;

#[cfg(all(feature = "deepstream", target_os = "linux"))]
mod live_perception;

pub use application::entry;
