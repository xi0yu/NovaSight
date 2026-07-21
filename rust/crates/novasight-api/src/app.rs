use axum::Router;

use crate::{ApiState, routes};

pub fn build_router(state: ApiState) -> Router {
    routes::router().with_state(state)
}
