//! Continuous Atan response for one selected target.
//!
//! Atan is the response curve: it already supplies a nonlinear proportional
//! response and a continuously decreasing effective gain as the error grows.
//! The radial response multiplier owns how strongly that curve is allowed to
//! act as the normalized target error grows.

#[derive(Clone, Copy, Debug)]
pub(super) struct ContinuousAtanConfig {
    pub scale_counts: f64,
    pub response_scale: f64,
    pub response_boost: f64,
    pub response_curve_shape: f64,
    pub max_counts_per_update: f64,
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
        full_error_counts_x: f64,
        full_error_counts_y: f64,
    ) -> Option<ContinuousDemand> {
        if !self.valid() || !full_error_counts_x.is_finite() || !full_error_counts_y.is_finite() {
            return None;
        }

        let error_radius_counts = full_error_counts_x.hypot(full_error_counts_y);
        let normalized_error = error_radius_counts / self.scale_counts;
        let response_gain = self.response_scale
            * response_multiplier(
                normalized_error,
                self.response_boost,
                self.response_curve_shape,
            );
        let limit_counts = self.max_counts_per_update;
        let x = atan_response(full_error_counts_x, response_gain, self.scale_counts);
        let y = atan_response(full_error_counts_y, response_gain, self.scale_counts);
        if !x.is_finite() || !y.is_finite() || !limit_counts.is_finite() {
            return None;
        }
        Some(ContinuousDemand { x, y, limit_counts })
    }

    fn valid(self) -> bool {
        self.scale_counts.is_finite()
            && self.scale_counts > 0.0
            && self.response_scale.is_finite()
            && self.response_scale >= 0.0
            && self.response_boost.is_finite()
            && self.response_boost >= 0.0
            && self.response_curve_shape.is_finite()
            && (0.5..=4.0).contains(&self.response_curve_shape)
            && valid_limit(self.max_counts_per_update)
    }
}

fn valid_limit(value: f64) -> bool {
    value.is_finite() && (1.0..=f64::from(i16::MAX)).contains(&value)
}

fn atan_response(error_counts: f64, kp: f64, scale_counts: f64) -> f64 {
    kp * scale_counts * (error_counts / scale_counts).atan()
}

fn response_multiplier(normalized_error: f64, boost: f64, shape: f64) -> f64 {
    let r = normalized_error.max(0.0);
    let gamma = shape.clamp(0.5, 4.0);
    1.0 + boost * (1.0 - (-r.powf(gamma)).exp())
}

#[cfg(test)]
mod tests {
    use super::{ContinuousAtanConfig, atan_response, response_multiplier};

    fn config() -> ContinuousAtanConfig {
        ContinuousAtanConfig {
            scale_counts: 256.0,
            response_scale: 0.20,
            response_boost: 0.50,
            response_curve_shape: 1.0,
            max_counts_per_update: 127.0,
        }
    }

    #[test]
    fn response_uses_base_gain_at_zero_error_and_single_limit() {
        let config = config();
        let response = config.evaluate(0.0, 0.0).expect("response");

        assert_eq!(response.x, atan_response(0.0, 0.20, config.scale_counts));
        assert_eq!(response.y, atan_response(0.0, 0.20, config.scale_counts));
        assert_eq!(response.limit_counts, config.max_counts_per_update);
    }

    #[test]
    fn response_multiplier_is_continuous_and_boosted() {
        let config = config();
        let small = config.evaluate(32.0, 0.0).expect("small");
        let large = config.evaluate(2_048.0, 0.0).expect("large");

        assert!(small.x > atan_response(32.0, config.response_scale, config.scale_counts));
        assert!(large.x > small.x);
        assert_eq!(small.limit_counts, config.max_counts_per_update);
        assert_eq!(large.limit_counts, config.max_counts_per_update);
    }

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
