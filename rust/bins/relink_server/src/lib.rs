//! NovaSight's unique production composition root.

mod application;
#[cfg(all(feature = "tensorrt", any(test, target_os = "linux")))]
mod model_contract;
mod server;

#[cfg(all(feature = "deepstream", target_os = "linux"))]
mod live_perception;

pub use application::entry;
