use thiserror::Error;

/// Errors raised while admitting or transitioning typed core-domain values.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum AppError {
    #[error("detection {object_id} has invalid {field}: {reason}")]
    InvalidDetection {
        object_id: u64,
        field: &'static str,
        reason: &'static str,
    },

    #[error("coordinate space must be non-zero, got {width}x{height}")]
    InvalidCoordinateSpace { width: u32, height: u32 },

    #[error("detection batch contains {actual} candidates; maximum is {maximum}")]
    TooManyDetections { actual: usize, maximum: usize },

    #[error("detection batch contains duplicate object ID {object_id}")]
    DuplicateObjectId { object_id: u64 },

    #[error(
        "detection {object_id} box ({x}, {y})..({right}, {bottom}) is outside coordinate space {width}x{height}"
    )]
    CoordinateSpaceMismatch {
        object_id: u64,
        x: f64,
        y: f64,
        right: f64,
        bottom: f64,
        width: u32,
        height: u32,
    },

    #[error("selected target does not belong to the supplied detection batch")]
    TargetBatchMismatch,

    #[error("replay controller gain must be finite and non-negative")]
    InvalidReplayGain,

    #[error("control timestamp {control_nanos} precedes frame timestamp {frame_nanos}")]
    NonMonotonicControlTime {
        frame_nanos: u64,
        control_nanos: u64,
    },

    #[error("replay controller output is outside signed 32-bit device-count range")]
    DeviceCountOutOfRange,
}
