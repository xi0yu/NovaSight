//! Pure continuous aim-control mathematics.
//!
//! This module owns the complete numeric path from measured pixel error and
//! predicted target displacement to continuous device-count demand. It has no
//! knowledge of targets, triggers, pipeline generations, telemetry, or device
//! delivery.

pub const DEFAULT_ATAN_SCALE_COUNTS: f64 = 256.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AxisPair {
    pub x: f64,
    pub y: f64,
}

impl AxisPair {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    fn finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AimControlParameters {
    pub source_width: u32,
    pub roi_width: u32,
    pub roi_height: u32,
    pub observation_width: u32,
    pub observation_height: u32,
    pub projection_fov_x_deg: f64,
    pub projection_counts_per_360: f64,
    pub response_scale: f64,
    pub response_boost: f64,
    pub response_curve_shape: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AimControlInput {
    pub measured_error_px: AxisPair,
    pub predicted_offset_px: AxisPair,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AimControlResult {
    pub predicted_error_px: AxisPair,
    pub projected_error_counts: AxisPair,
    pub demand_counts: AxisPair,
    pub response_multiplier: f64,
    pub effective_gain: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct AimControlLaw {
    parameters: AimControlParameters,
    source_error_scale: AxisPair,
    counts_per_rad: f64,
}

impl AimControlLaw {
    pub fn new(parameters: AimControlParameters) -> Option<Self> {
        if !parameters_valid(parameters) {
            return None;
        }
        let fov_x_rad = parameters.projection_fov_x_deg.to_radians();
        let focal_px = (f64::from(parameters.source_width) * 0.5) / (fov_x_rad * 0.5).tan();
        if !focal_px.is_finite() || focal_px <= 0.0 {
            return None;
        }
        Some(Self {
            parameters,
            source_error_scale: AxisPair::new(
                f64::from(parameters.roi_width)
                    / f64::from(parameters.observation_width)
                    / focal_px,
                f64::from(parameters.roi_height)
                    / f64::from(parameters.observation_height)
                    / focal_px,
            ),
            counts_per_rad: parameters.projection_counts_per_360 / std::f64::consts::TAU,
        })
    }

    pub fn evaluate(self, input: AimControlInput) -> Option<AimControlResult> {
        if !input.measured_error_px.finite() || !input.predicted_offset_px.finite() {
            return None;
        }

        let predicted_error_px = input.measured_error_px.add(input.predicted_offset_px);
        let projected_error_counts = AxisPair::new(
            (predicted_error_px.x * self.source_error_scale.x).atan() * self.counts_per_rad,
            (predicted_error_px.y * self.source_error_scale.y).atan() * self.counts_per_rad,
        );
        let normalized_error =
            projected_error_counts.x.hypot(projected_error_counts.y) / DEFAULT_ATAN_SCALE_COUNTS;
        let multiplier = response_multiplier(
            normalized_error,
            self.parameters.response_boost,
            self.parameters.response_curve_shape,
        );
        let effective_gain = self.parameters.response_scale * multiplier;
        let demand_counts = AxisPair::new(
            atan_response(projected_error_counts.x, effective_gain),
            atan_response(projected_error_counts.y, effective_gain),
        );
        if !projected_error_counts.finite()
            || !demand_counts.finite()
            || !effective_gain.is_finite()
        {
            return None;
        }

        Some(AimControlResult {
            predicted_error_px,
            projected_error_counts,
            demand_counts,
            response_multiplier: multiplier,
            effective_gain,
        })
    }
}

fn parameters_valid(parameters: AimControlParameters) -> bool {
    parameters.source_width > 0
        && parameters.roi_width > 0
        && parameters.roi_height > 0
        && parameters.observation_width > 0
        && parameters.observation_height > 0
        && parameters.projection_fov_x_deg.is_finite()
        && (0.0..180.0).contains(&parameters.projection_fov_x_deg)
        && parameters.projection_counts_per_360.is_finite()
        && parameters.projection_counts_per_360 > 0.0
        && parameters.response_scale.is_finite()
        && parameters.response_scale >= 0.0
        && parameters.response_boost.is_finite()
        && parameters.response_boost >= 0.0
        && parameters.response_curve_shape.is_finite()
        && (0.5..=4.0).contains(&parameters.response_curve_shape)
}

fn atan_response(error_counts: f64, gain: f64) -> f64 {
    gain * DEFAULT_ATAN_SCALE_COUNTS * (error_counts / DEFAULT_ATAN_SCALE_COUNTS).atan()
}

fn response_multiplier(normalized_error: f64, boost: f64, shape: f64) -> f64 {
    let radius = normalized_error.max(0.0);
    let gamma = shape.clamp(0.5, 4.0);
    1.0 + boost * (1.0 - (-radius.powf(gamma)).exp())
}

#[cfg(test)]
mod tests {
    use super::response_multiplier;

    #[test]
    fn response_multiplier_approaches_configured_boost() {
        let boosted = response_multiplier(64.0, 0.5, 1.0);
        assert!((boosted - 1.5).abs() < 1e-12);
    }

    #[test]
    fn response_shape_adjusts_transition_without_leaving_bounds() {
        let early = response_multiplier(0.25, 0.5, 0.5);
        let neutral = response_multiplier(0.25, 0.5, 1.0);
        let late = response_multiplier(0.25, 0.5, 2.0);

        assert!(early > neutral);
        assert!(neutral > late);
        assert!((1.0..=1.5).contains(&early));
        assert!((1.0..=1.5).contains(&late));
    }
}
