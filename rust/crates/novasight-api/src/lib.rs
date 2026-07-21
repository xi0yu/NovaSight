#![forbid(unsafe_code)]

mod app;
pub mod dto;
mod error;
mod routes;
mod state;

pub use app::build_router;
pub use state::ApiState;
