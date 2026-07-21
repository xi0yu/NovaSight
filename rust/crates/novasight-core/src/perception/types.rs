use serde::{Deserialize, Serialize};

use crate::AppError;

/// Hard admission bound preventing unbounded per-frame candidate work.
pub const MAX_DETECTIONS: usize = 256;

/// Identifies one runtime session; values from different epochs must never mix.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct RuntimeEpoch(pub u64);

/// Monotonic observation generation within a runtime epoch.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Generation(pub u64);

impl PartialEq<u64> for Generation {
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Generation> for u64 {
    fn eq(&self, other: &Generation) -> bool {
        *self == other.0
    }
}

/// Monotonic clock value in nanoseconds, never wall-clock time.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct MonotonicNanos(pub u64);

/// Identity and capture time attached to one admitted perception frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FrameStamp {
    pub epoch: RuntimeEpoch,
    pub generation: Generation,
    pub captured_at: MonotonicNanos,
}

impl FrameStamp {
    pub const fn new(epoch: RuntimeEpoch, generation: u64, captured_at_nanos: u64) -> Self {
        Self {
            epoch,
            generation: Generation(generation),
            captured_at: MonotonicNanos(captured_at_nanos),
        }
    }
}

/// One finite bounding box in the batch's declared coordinate space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    object_id: u64,
    class_id: u32,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    confidence: f32,
}

impl Detection {
    /// Validates all numeric fields before a detection can enter targeting.
    pub fn new(
        object_id: u64,
        class_id: u32,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        confidence: f32,
    ) -> Result<Self, AppError> {
        for (field, value) in [
            ("x", x),
            ("y", y),
            ("width", width),
            ("height", height),
            ("confidence", confidence),
        ] {
            if !value.is_finite() {
                return Err(AppError::InvalidDetection {
                    object_id,
                    field,
                    reason: "must be finite",
                });
            }
        }
        if width <= 0.0 {
            return Err(AppError::InvalidDetection {
                object_id,
                field: "width",
                reason: "must be positive",
            });
        }
        if height <= 0.0 {
            return Err(AppError::InvalidDetection {
                object_id,
                field: "height",
                reason: "must be positive",
            });
        }
        if !(0.0..=1.0).contains(&confidence) {
            return Err(AppError::InvalidDetection {
                object_id,
                field: "confidence",
                reason: "must be within 0.0..=1.0",
            });
        }

        Ok(Self {
            object_id,
            class_id,
            x,
            y,
            width,
            height,
            confidence,
        })
    }

    pub const fn object_id(&self) -> u64 {
        self.object_id
    }

    pub const fn class_id(&self) -> u32 {
        self.class_id
    }

    pub const fn x(&self) -> f32 {
        self.x
    }

    pub const fn y(&self) -> f32 {
        self.y
    }

    pub const fn width(&self) -> f32 {
        self.width
    }

    pub const fn height(&self) -> f32 {
        self.height
    }

    pub const fn confidence(&self) -> f32 {
        self.confidence
    }

    pub fn center_x(&self) -> f32 {
        self.x + self.width * 0.5
    }

    pub fn center_y(&self) -> f32 {
        self.y + self.height * 0.5
    }
}

/// A bounded set of validated detections sharing one frame and coordinate space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectionBatch {
    stamp: FrameStamp,
    coordinate_width: u32,
    coordinate_height: u32,
    detections: Vec<Detection>,
}

impl DetectionBatch {
    /// Admits only non-zero geometry, at most [`MAX_DETECTIONS`] candidates,
    /// and detection centers within the declared coordinate space.
    pub fn new(
        stamp: FrameStamp,
        coordinate_width: u32,
        coordinate_height: u32,
        detections: Vec<Detection>,
    ) -> Result<Self, AppError> {
        if coordinate_width == 0 || coordinate_height == 0 {
            return Err(AppError::InvalidCoordinateSpace {
                width: coordinate_width,
                height: coordinate_height,
            });
        }
        if detections.len() > MAX_DETECTIONS {
            return Err(AppError::TooManyDetections {
                actual: detections.len(),
                maximum: MAX_DETECTIONS,
            });
        }

        let width = coordinate_width as f32;
        let height = coordinate_height as f32;
        for detection in &detections {
            let center_x = detection.center_x();
            let center_y = detection.center_y();
            if !(0.0..width).contains(&center_x) || !(0.0..height).contains(&center_y) {
                return Err(AppError::CoordinateSpaceMismatch {
                    object_id: detection.object_id,
                    center_x,
                    center_y,
                    width: coordinate_width,
                    height: coordinate_height,
                });
            }
        }

        Ok(Self {
            stamp,
            coordinate_width,
            coordinate_height,
            detections,
        })
    }

    /// Test/replay constructor with the same validation as production admission.
    pub fn fixture(
        stamp: FrameStamp,
        coordinate_width: u32,
        coordinate_height: u32,
        detections: Vec<Detection>,
    ) -> Result<Self, AppError> {
        Self::new(stamp, coordinate_width, coordinate_height, detections)
    }

    pub const fn stamp(&self) -> FrameStamp {
        self.stamp
    }

    pub const fn coordinate_width(&self) -> u32 {
        self.coordinate_width
    }

    pub const fn coordinate_height(&self) -> u32 {
        self.coordinate_height
    }

    pub fn detections(&self) -> &[Detection] {
        &self.detections
    }

    pub fn center(&self) -> (f32, f32) {
        (
            self.coordinate_width as f32 * 0.5,
            self.coordinate_height as f32 * 0.5,
        )
    }
}
