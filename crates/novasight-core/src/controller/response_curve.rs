//! Continuous Atan response for one selected target.
//!
//! Atan is the response curve: it already supplies a nonlinear proportional
//! response and a continuously decreasing effective gain as the error grows.
//! The radial response schedule owns how strongly that curve is allowed to act
//! as the predicted target distance moves from settled to far.

#[derive(Clone, Copy, Debug)]
pub(super) struct ContinuousAtanConfig {
    pub response_curve_center_px: f64,
    pub response_curve_width_ratio: f64,
    pub response_curve_shape: f64,
    pub scale_counts: f64,
    pub response_scale: f64,
    pub response_gain_floor: f64,
    pub response_gain_ceiling: f64,
    pub near_limit_counts: f64,
    pub far_limit_counts: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ContinuousDemand {
    pub x: f64,
    pub y: f64,
    pub limit_counts: f64,
}

impl ContinuousAtanConfig {
    pub fn evaluate(
        self,
        measured_distance_px: f64,
        full_error_counts_x: f64,
        full_error_counts_y: f64,
    ) -> Option<ContinuousDemand> {
        if !self.valid()
            || !measured_distance_px.is_finite()
            || measured_distance_px < 0.0
            || !full_error_counts_x.is_finite()
            || !full_error_counts_y.is_finite()
        {
            return None;
        }

        let progress = response_progress(
            measured_distance_px,
            self.response_curve_center_px,
            self.response_curve_width_ratio,
            self.response_curve_shape,
        );
        let response_gain = self.response_scale
            * lerp(
                self.response_gain_floor,
                self.response_gain_ceiling,
                progress,
            );
        let limit_counts = lerp(self.near_limit_counts, self.far_limit_counts, progress);
        let x = atan_response(full_error_counts_x, response_gain, self.scale_counts);
        let y = atan_response(full_error_counts_y, response_gain, self.scale_counts);
        if !x.is_finite() || !y.is_finite() || !limit_counts.is_finite() {
            return None;
        }
        Some(ContinuousDemand { x, y, limit_counts })
    }

    fn valid(self) -> bool {
        self.response_curve_center_px.is_finite()
            && self.response_curve_center_px >= 0.0
            && self.response_curve_width_ratio.is_finite()
            && self.response_curve_width_ratio > 0.0
            && self.response_curve_shape.is_finite()
            && (0.5..=4.0).contains(&self.response_curve_shape)
            && self.scale_counts.is_finite()
            && self.scale_counts > 0.0
            && self.response_scale.is_finite()
            && self.response_scale >= 0.0
            && self.response_gain_floor.is_finite()
            && self.response_gain_floor >= 0.0
            && self.response_gain_ceiling.is_finite()
            && self.response_gain_floor <= self.response_gain_ceiling
            && valid_limit(self.far_limit_counts)
            && valid_limit(self.near_limit_counts)
    }
}

fn valid_limit(value: f64) -> bool {
    value.is_finite() && (1.0..=f64::from(i16::MAX)).contains(&value)
}

fn atan_response(error_counts: f64, kp: f64, scale_counts: f64) -> f64 {
    kp * scale_counts * (error_counts / scale_counts).atan()
}

/// Return a bounded radial response progress value. With the default width and
/// shape it exactly matches the legacy smoothstep schedule used by migrations.
pub(super) fn response_progress(
    distance_px: f64,
    center_px: f64,
    width_ratio: f64,
    shape: f64,
) -> f64 {
    if center_px <= f64::EPSILON {
        return 1.0;
    }
    let half_width = (center_px * width_ratio.max(1e-9)).max(1e-9);
    let inner = center_px - half_width;
    let outer = center_px + half_width;
    let t = ((distance_px - inner) / (outer - inner)).clamp(0.0, 1.0);
    shaped_smoothstep(t, shape)
}

fn shaped_smoothstep(t: f64, shape: f64) -> f64 {
    let base = t * t * (3.0 - 2.0 * t);
    if (shape - 1.0).abs() <= f64::EPSILON {
        return base;
    }
    let gamma = shape.clamp(0.5, 4.0);
    if base <= 0.0 || base >= 1.0 {
        return base;
    }
    let left = base.powf(gamma);
    let right = (1.0 - base).powf(gamma);
    left / (left + right)
}

fn lerp(start: f64, end: f64, weight: f64) -> f64 {
    start + (end - start) * weight
}

#[cfg(test)]
mod tests {
    use super::{ContinuousAtanConfig, atan_response, response_progress};

    const DEFAULT_RESPONSE_CURVE_WIDTH_RATIO: f64 = 0.25;
    const DEFAULT_RESPONSE_CURVE_SHAPE: f64 = 1.0;

    fn config() -> ContinuousAtanConfig {
        ContinuousAtanConfig {
            response_curve_center_px: 12.0,
            response_curve_width_ratio: DEFAULT_RESPONSE_CURVE_WIDTH_RATIO,
            response_curve_shape: DEFAULT_RESPONSE_CURVE_SHAPE,
            scale_counts: 256.0,
            response_scale: 0.22,
            response_gain_floor: 0.20 / 0.22,
            response_gain_ceiling: 1.0,
            near_limit_counts: 72.0,
            far_limit_counts: 127.0,
        }
    }

    #[test]
    fn transition_preserves_exact_near_and_far_responses_outside_its_band() {
        let config = config();
        let full_error = 80.0;
        let near = config.evaluate(8.0, full_error, 0.0).expect("near");
        let far = config.evaluate(16.0, full_error, 0.0).expect("far");

        assert_eq!(near.x, atan_response(full_error, 0.20, config.scale_counts));
        assert_eq!(far.x, atan_response(full_error, 0.22, config.scale_counts));
        assert_eq!(near.limit_counts, config.near_limit_counts);
        assert_eq!(far.limit_counts, config.far_limit_counts);
    }

    #[test]
    fn transition_is_continuous_at_the_configured_center() {
        let config = config();
        let before = config.evaluate(12.0 - 1e-6, 80.0, -40.0).expect("before");
        let after = config.evaluate(12.0 + 1e-6, 80.0, -40.0).expect("after");

        assert!((after.x - before.x).abs() < 1e-5);
        assert!((after.y - before.y).abs() < 1e-5);
        assert!((after.limit_counts - before.limit_counts).abs() < 1e-4);
    }

    #[test]
    fn migrated_profile_exactly_matches_legacy_gain_schedule() {
        let config = config();
        for distance in [0.0, 6.0, 9.0, 12.0, 15.0, 18.0, 24.0] {
            let progress = response_progress(
                distance,
                config.response_curve_center_px,
                config.response_curve_width_ratio,
                config.response_curve_shape,
            );
            let legacy_kp = 0.20 + (0.22 - 0.20) * progress;
            let migrated_gain = config.response_scale
                * (config.response_gain_floor
                    + (config.response_gain_ceiling - config.response_gain_floor) * progress);
            assert!((migrated_gain - legacy_kp).abs() < 1e-12);
        }
    }

    #[test]
    fn response_shape_adjusts_transition_without_leaving_bounds() {
        let early = response_progress(10.0, 12.0, DEFAULT_RESPONSE_CURVE_WIDTH_RATIO, 0.5);
        let neutral = response_progress(10.0, 12.0, DEFAULT_RESPONSE_CURVE_WIDTH_RATIO, 1.0);
        let late = response_progress(10.0, 12.0, DEFAULT_RESPONSE_CURVE_WIDTH_RATIO, 2.0);

        assert!(early > neutral);
        assert!(neutral > late);
        assert!((0.0..=1.0).contains(&early));
        assert!((0.0..=1.0).contains(&late));
    }
}
