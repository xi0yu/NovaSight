//! Continuous FAR/NEAR Atan response for one selected target.
//!
//! Atan is the response curve: it already supplies a nonlinear proportional
//! response and a continuously decreasing effective gain as the error grows.
//! FAR and NEAR are therefore blended as two parameterizations of that one
//! curve instead of being stacked as an additional gain stage.

const TRANSITION_HALF_WIDTH_RATIO: f64 = 0.25;

#[derive(Clone, Copy, Debug)]
pub(super) struct BlendedAtanConfig {
    pub near_threshold_px: f64,
    pub scale_counts: f64,
    pub far_kp: f64,
    pub far_limit_counts: f64,
    pub near_kp: f64,
    pub near_limit_counts: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ContinuousDemand {
    pub x: f64,
    pub y: f64,
    pub limit_counts: f64,
}

impl BlendedAtanConfig {
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

        let far_weight = far_weight(measured_distance_px, self.near_threshold_px);
        if far_weight <= 0.0 {
            return single_region_demand(
                full_error_counts_x,
                full_error_counts_y,
                self.near_kp,
                self.scale_counts,
                self.near_limit_counts,
            );
        }
        if far_weight >= 1.0 {
            return single_region_demand(
                full_error_counts_x,
                full_error_counts_y,
                self.far_kp,
                self.scale_counts,
                self.far_limit_counts,
            );
        }
        let near_x = atan_response(full_error_counts_x, self.near_kp, self.scale_counts);
        let near_y = atan_response(full_error_counts_y, self.near_kp, self.scale_counts);
        let far_x = atan_response(full_error_counts_x, self.far_kp, self.scale_counts);
        let far_y = atan_response(full_error_counts_y, self.far_kp, self.scale_counts);
        let limit_counts = lerp(self.near_limit_counts, self.far_limit_counts, far_weight);
        let x = lerp(near_x, far_x, far_weight);
        let y = lerp(near_y, far_y, far_weight);
        if !x.is_finite() || !y.is_finite() || !limit_counts.is_finite() {
            return None;
        }
        Some(ContinuousDemand { x, y, limit_counts })
    }

    fn valid(self) -> bool {
        self.near_threshold_px.is_finite()
            && self.near_threshold_px >= 0.0
            && self.scale_counts.is_finite()
            && self.scale_counts > 0.0
            && self.far_kp.is_finite()
            && self.far_kp >= 0.0
            && self.near_kp.is_finite()
            && self.near_kp >= 0.0
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

fn single_region_demand(
    full_error_counts_x: f64,
    full_error_counts_y: f64,
    kp: f64,
    scale_counts: f64,
    limit_counts: f64,
) -> Option<ContinuousDemand> {
    let x = atan_response(full_error_counts_x, kp, scale_counts);
    let y = atan_response(full_error_counts_y, kp, scale_counts);
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    Some(ContinuousDemand { x, y, limit_counts })
}

/// FAR and NEAR remain exact outside a transition band spanning 50% of the
/// configured threshold. Inside it, cubic smoothstep removes the parameter
/// jump without introducing another gain stage or another user-facing knob.
fn far_weight(distance_px: f64, near_threshold_px: f64) -> f64 {
    if near_threshold_px <= f64::EPSILON {
        return 1.0;
    }
    let half_width = near_threshold_px * TRANSITION_HALF_WIDTH_RATIO;
    let inner = near_threshold_px - half_width;
    let outer = near_threshold_px + half_width;
    let t = ((distance_px - inner) / (outer - inner)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(start: f64, end: f64, weight: f64) -> f64 {
    start + (end - start) * weight
}

#[cfg(test)]
mod tests {
    use super::{BlendedAtanConfig, atan_response};

    fn config() -> BlendedAtanConfig {
        BlendedAtanConfig {
            near_threshold_px: 12.0,
            scale_counts: 256.0,
            far_kp: 0.22,
            far_limit_counts: 127.0,
            near_kp: 0.20,
            near_limit_counts: 72.0,
        }
    }

    #[test]
    fn transition_preserves_exact_near_and_far_responses_outside_its_band() {
        let config = config();
        let full_error = 80.0;
        let near = config.evaluate(8.0, full_error, 0.0).expect("near");
        let far = config.evaluate(16.0, full_error, 0.0).expect("far");

        assert_eq!(
            near.x,
            atan_response(full_error, config.near_kp, config.scale_counts)
        );
        assert_eq!(
            far.x,
            atan_response(full_error, config.far_kp, config.scale_counts)
        );
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
}
