//! Typed units used by the Phase 2 coordinate geometry.
//!
//! Each newtype owns a single, non-negative, finite pixel count. Newtypes
//! are deliberately thin: they enforce the validation rules once at the
//! boundary so downstream geometry code never has to re-check finiteness,
//! sign, or non-zero dimensions. Conversions between spaces are explicit
//! and live in the geometry transform layer; this module only knows about
//! validated pixel values.

use serde::{Deserialize, Serialize};

use crate::error::AppError;

/// Marker trait shared by every validated pixel newtype.
pub trait PixelCount {
    /// Underlying `f64` value; always non-negative and finite.
    fn value(self) -> f64;
}

/// Generic validated width container used by all spaces.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pixels<Tag> {
    value: f64,
    _tag: core::marker::PhantomData<Tag>,
}

impl<Tag> Pixels<Tag> {
    /// Validate and construct a width or height.
    pub fn new(value: f64) -> Result<Self, AppError> {
        if !value.is_finite() {
            return Err(AppError::InvalidCoordinateSpace {
                width: f64::NAN.to_bits() as u32,
                height: f64::NAN.to_bits() as u32,
            });
        }
        if value <= 0.0 {
            return Err(AppError::InvalidCoordinateSpace {
                width: 0,
                height: 0,
            });
        }
        Ok(Self {
            value,
            _tag: core::marker::PhantomData,
        })
    }

    /// Borrow the underlying validated value.
    pub const fn get(self) -> f64 {
        self.value
    }
}

impl<Tag> PixelCount for Pixels<Tag> {
    fn value(self) -> f64 {
        self.value
    }
}

/// Coordinate-space tags. Distinct types prevent the compiler from mixing
/// model, ROI, capture, control, and display widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelSpace {}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoiSpace {}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureSpace {}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlSpace {}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplaySpace {}

pub type ModelPixels = Pixels<ModelSpace>;
pub type RoiPixels = Pixels<RoiSpace>;
pub type CapturePixels = Pixels<CaptureSpace>;
pub type ControlPixels = Pixels<ControlSpace>;
pub type DisplayPixels = Pixels<DisplaySpace>;

/// Validated non-negative coordinate value (top-left origin or relative offset).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Coordinate<Tag> {
    value: f64,
    _tag: core::marker::PhantomData<Tag>,
}

impl<Tag> Coordinate<Tag> {
    pub fn new(value: f64) -> Result<Self, AppError> {
        if !value.is_finite() {
            return Err(AppError::InvalidCoordinateSpace {
                width: f64::NAN.to_bits() as u32,
                height: f64::NAN.to_bits() as u32,
            });
        }
        Ok(Self {
            value,
            _tag: core::marker::PhantomData,
        })
    }

    pub const fn get(self) -> f64 {
        self.value
    }
}

impl<Tag> Default for Coordinate<Tag> {
    fn default() -> Self {
        Self {
            value: 0.0,
            _tag: core::marker::PhantomData,
        }
    }
}

pub type ModelCoordinate = Coordinate<ModelSpace>;
pub type RoiCoordinate = Coordinate<RoiSpace>;
pub type CaptureCoordinate = Coordinate<CaptureSpace>;
pub type ControlCoordinate = Coordinate<ControlSpace>;
pub type DisplayCoordinate = Coordinate<DisplaySpace>;

/// Validated non-negative duration in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Nanoseconds(pub u64);

/// Validated strictly-positive scale factor. Display/control round-trips
/// require both scale and its reciprocal to be finite and non-zero.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scale(pub f64);

impl Scale {
    pub fn new(value: f64) -> Result<Self, AppError> {
        if !value.is_finite() || value <= 0.0 {
            return Err(AppError::InvalidCoordinateSpace {
                width: 0,
                height: 0,
            });
        }
        Ok(Self(value))
    }
}
