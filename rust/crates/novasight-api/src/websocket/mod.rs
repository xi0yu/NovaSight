pub mod status;

use axum::Router;

use crate::ApiState;

pub(crate) fn router() -> Router<ApiState> {
    status::router()
}
