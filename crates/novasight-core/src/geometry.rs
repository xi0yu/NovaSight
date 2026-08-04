//! Current coordinate-space transform.
//!
//! Mirrors the semantics of ``novasight.coordinates.CoordinateTransform``
//! without depending on any Python, OpenCV, or GStreamer type. Every
//! transform is an explicit round trip in `f64`; rounding to integer
//! pixel values happens only at named boundaries (typed `u32` outputs)
//! where callers request an integer conversion.

use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::units::{
    CaptureCoordinate, CapturePixels, ControlCoordinate, DisplayCoordinate, ModelCoordinate,
    ModelPixels, RoiCoordinate, RoiPixels, Scale,
};

/// Pixel point in any space. Values are `f64`; rounding is the caller's job.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Axis-aligned bounding box in any space.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

impl BBox {
    pub const fn from_xyxy(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self { x1, y1, x2, y2 }
    }

    pub const fn width(&self) -> f64 {
        self.x2 - self.x1
    }

    pub const fn height(&self) -> f64 {
        self.y2 - self.y1
    }

    pub const fn center(&self) -> Point {
        Point::new((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }
}

/// Source-of-truth transform parameters. All dimensions are validated and
/// in the typed ``Pixels`` newtype. Callers build it from a runtime frame
/// stamp and persist it on every typed observation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoordinateTransform {
    model_width: ModelPixels,
    model_height: ModelPixels,
    roi_x: RoiCoordinate,
    roi_y: RoiCoordinate,
    roi_width: RoiPixels,
    roi_height: RoiPixels,
    capture_width: CapturePixels,
    capture_height: CapturePixels,
    control_origin_x: ControlCoordinate,
    control_origin_y: ControlCoordinate,
    display_scale_x: Scale,
    display_scale_y: Scale,
}

impl CoordinateTransform {
    /// Build a transform from validated typed units. Returns the first
    /// invalid argument as an `AppError::InvalidCoordinateSpace`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_width: ModelPixels,
        model_height: ModelPixels,
        roi_x: RoiCoordinate,
        roi_y: RoiCoordinate,
        roi_width: RoiPixels,
        roi_height: RoiPixels,
        capture_width: CapturePixels,
        capture_height: CapturePixels,
        control_origin_x: ControlCoordinate,
        control_origin_y: ControlCoordinate,
        display_scale_x: Scale,
        display_scale_y: Scale,
    ) -> Result<Self, AppError> {
        Ok(Self {
            model_width,
            model_height,
            roi_x,
            roi_y,
            roi_width,
            roi_height,
            capture_width,
            capture_height,
            control_origin_x,
            control_origin_y,
            display_scale_x,
            display_scale_y,
        })
    }

    pub fn model_to_roi_scale_x(&self) -> f64 {
        self.roi_width.get() / self.model_width.get()
    }

    pub fn model_to_roi_scale_y(&self) -> f64 {
        self.roi_height.get() / self.model_height.get()
    }

    pub fn model_to_roi_point(&self, x: f64, y: f64) -> Point {
        Point::new(
            x * self.model_to_roi_scale_x(),
            y * self.model_to_roi_scale_y(),
        )
    }

    pub fn roi_to_model_point(&self, x: f64, y: f64) -> Point {
        Point::new(
            x / self.model_to_roi_scale_x(),
            y / self.model_to_roi_scale_y(),
        )
    }

    pub fn roi_to_capture_point(&self, x: f64, y: f64) -> Point {
        Point::new(x + self.roi_x.get(), y + self.roi_y.get())
    }

    pub fn capture_to_roi_point(&self, x: f64, y: f64) -> Point {
        Point::new(x - self.roi_x.get(), y - self.roi_y.get())
    }

    pub fn capture_to_control_point(&self, x: f64, y: f64) -> Point {
        Point::new(
            x - self.control_origin_x.get(),
            y - self.control_origin_y.get(),
        )
    }

    pub fn control_to_capture_point(&self, x: f64, y: f64) -> Point {
        Point::new(
            x + self.control_origin_x.get(),
            y + self.control_origin_y.get(),
        )
    }

    pub fn control_to_display_point(&self, x: f64, y: f64) -> Point {
        Point::new(x * self.display_scale_x.0, y * self.display_scale_y.0)
    }

    pub fn display_to_control_point(&self, x: f64, y: f64) -> Point {
        Point::new(x / self.display_scale_x.0, y / self.display_scale_y.0)
    }

    pub fn model_to_roi_box(&self, bbox: BBox) -> BBox {
        let top_left = self.model_to_roi_point(bbox.x1, bbox.y1);
        let bottom_right = self.model_to_roi_point(bbox.x2, bbox.y2);
        BBox::from_xyxy(top_left.x, top_left.y, bottom_right.x, bottom_right.y)
    }

    pub fn roi_to_capture_box(&self, bbox: BBox) -> BBox {
        let top_left = self.roi_to_capture_point(bbox.x1, bbox.y1);
        let bottom_right = self.roi_to_capture_point(bbox.x2, bbox.y2);
        BBox::from_xyxy(top_left.x, top_left.y, bottom_right.x, bottom_right.y)
    }

    pub fn capture_to_control_box(&self, bbox: BBox) -> BBox {
        let top_left = self.capture_to_control_point(bbox.x1, bbox.y1);
        let bottom_right = self.capture_to_control_point(bbox.x2, bbox.y2);
        BBox::from_xyxy(top_left.x, top_left.y, bottom_right.x, bottom_right.y)
    }

    pub fn control_to_display_box(&self, bbox: BBox) -> BBox {
        let top_left = self.control_to_display_point(bbox.x1, bbox.y1);
        let bottom_right = self.control_to_display_point(bbox.x2, bbox.y2);
        BBox::from_xyxy(top_left.x, top_left.y, bottom_right.x, bottom_right.y)
    }

    pub fn roi_to_control_box(&self, bbox: BBox) -> BBox {
        self.capture_to_control_box(self.roi_to_capture_box(bbox))
    }

    pub fn roi_to_display_box(&self, bbox: BBox) -> BBox {
        self.control_to_display_box(self.roi_to_control_box(bbox))
    }

    pub fn model_to_control_box(&self, bbox: BBox) -> BBox {
        self.capture_to_control_box(self.roi_to_capture_box(self.model_to_roi_box(bbox)))
    }
}

/// Convenience constructor for the common replay fixture shape.
pub fn standard_transform(
    model: f64,
    roi: f64,
    capture: f64,
) -> Result<CoordinateTransform, AppError> {
    CoordinateTransform::new(
        ModelPixels::new(model)?,
        ModelPixels::new(model)?,
        RoiCoordinate::new(0.0)?,
        RoiCoordinate::new(0.0)?,
        RoiPixels::new(roi)?,
        RoiPixels::new(roi)?,
        CapturePixels::new(capture)?,
        CapturePixels::new(capture)?,
        ControlCoordinate::new(0.0)?,
        ControlCoordinate::new(0.0)?,
        Scale::new(1.0)?,
        Scale::new(1.0)?,
    )
}

// Silence unused warnings when the geometry module is consumed but the
// re-exports are not all reached in a given build.
#[allow(dead_code)]
fn _type_re_exports() -> (
    ModelCoordinate,
    RoiCoordinate,
    CaptureCoordinate,
    ControlCoordinate,
    DisplayCoordinate,
) {
    (
        ModelCoordinate::new(0.0).unwrap(),
        RoiCoordinate::new(0.0).unwrap(),
        CaptureCoordinate::new(0.0).unwrap(),
        ControlCoordinate::new(0.0).unwrap(),
        DisplayCoordinate::new(0.0).unwrap(),
    )
}
