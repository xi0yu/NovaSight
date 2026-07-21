use axum::Router;

use crate::{ApiState, routes, websocket};

pub fn build_router(state: ApiState) -> Router {
    routes::router()
        .merge(websocket::router())
        .with_state(state)
}
