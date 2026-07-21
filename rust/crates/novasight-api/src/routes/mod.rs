mod compat;
mod health;
mod runtime;

use axum::Router;

use crate::ApiState;

pub(crate) fn router() -> Router<ApiState> {
    Router::new()
        .merge(health::router())
        .merge(runtime::router())
        .merge(compat::router())
}
