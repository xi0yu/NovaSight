use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use novasight_core::AppError;

use crate::dto::ErrorResponse;

pub(crate) type ApiResult<T> = Result<T, ApiError>;

pub(crate) struct ApiError(AppError);

impl From<AppError> for ApiError {
    fn from(error: AppError) -> Self {
        Self(error)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = status_for(&self.0);
        let body = ErrorResponse::new(code_for(&self.0), self.0.to_string());
        (status, Json(body)).into_response()
    }
}

fn status_for(error: &AppError) -> StatusCode {
    match error {
        AppError::RuntimeCommandConflict { .. } | AppError::RuntimeEpochMismatch { .. } => {
            StatusCode::CONFLICT
        }
        AppError::RuntimeManagerUnavailable | AppError::RuntimeTaskTerminated => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        AppError::RuntimeEpochExhausted => StatusCode::INTERNAL_SERVER_ERROR,
        AppError::InvalidDetection { .. }
        | AppError::InvalidCoordinateSpace { .. }
        | AppError::TooManyDetections { .. }
        | AppError::DuplicateObjectId { .. }
        | AppError::CoordinateSpaceMismatch { .. }
        | AppError::TargetBatchMismatch
        | AppError::InvalidReplayGain
        | AppError::NonMonotonicControlTime { .. }
        | AppError::DeviceCountOutOfRange => StatusCode::BAD_REQUEST,
    }
}

fn code_for(error: &AppError) -> &'static str {
    match error {
        AppError::InvalidDetection { .. } => "invalid_detection",
        AppError::InvalidCoordinateSpace { .. } => "invalid_coordinate_space",
        AppError::TooManyDetections { .. } => "too_many_detections",
        AppError::DuplicateObjectId { .. } => "duplicate_object_id",
        AppError::CoordinateSpaceMismatch { .. } => "coordinate_space_mismatch",
        AppError::TargetBatchMismatch => "target_batch_mismatch",
        AppError::InvalidReplayGain => "invalid_replay_gain",
        AppError::NonMonotonicControlTime { .. } => "non_monotonic_control_time",
        AppError::DeviceCountOutOfRange => "device_count_out_of_range",
        AppError::RuntimeCommandConflict { .. } => "runtime_command_conflict",
        AppError::RuntimeEpochMismatch { .. } => "runtime_epoch_mismatch",
        AppError::RuntimeEpochExhausted => "runtime_epoch_exhausted",
        AppError::RuntimeManagerUnavailable => "runtime_manager_unavailable",
        AppError::RuntimeTaskTerminated => "runtime_task_terminated",
    }
}
