#![forbid(unsafe_code)]

mod app;
mod control;
pub mod dto;
mod error;
mod routes;
mod state;
pub mod websocket;

pub use app::build_router;
pub use control::{
    build_control_router, build_control_router_with_capabilities,
    build_control_router_with_control_plane, build_control_router_with_platform_queries,
    build_control_router_with_services, build_control_router_with_shutdown,
};
pub use state::ApiState;
