//! Short-lived vertical correction at the device output seam.
//!
//! Only confirmed sends enter the response estimate. The module never changes
//! target selection, prediction, or the continuous Atan aim law.

use std::collections::VecDeque;

use novasight_core::controller::{AimAlgorithmConfig, FireStabilizationConfig};

const MAX_PENDING_SENDS: usize = 32;
const MAX_OBSERVATION_GAP_NS: u64 = 80_000_000;
const RESIDUAL_NOISE_PX: f64 = 0.4;
const OVERSHOOT_PX: f64 = 3.0;

pub(super) fn pixels_per_count_y(config: AimAlgorithmConfig, observation_height: u32) -> f64 {
    if config.source_width == 0
        || config.roi_height == 0
        || observation_height == 0
        || config.projection_counts_per_360 <= 0.0
        || !(0.0..180.0).contains(&config.projection_fov_x_deg)
    {
        return 0.0;
    }
    let focal_px = f64::from(config.source_width) * 0.5
        / (config.projection_fov_x_deg.to_radians() * 0.5).tan();
    f64::from(observation_height) / f64::from(config.roi_height) * focal_px * std::f64::consts::TAU
        / config.projection_counts_per_360
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FireObservation {
    pub target_id: u64,
    pub capture_ns: u64,
    pub issued_ns: u64,
    pub error_x_px: f64,
    pub error_y_px: f64,
    pub target_reset: bool,
    pub left_held: bool,
    pub base_y_counts: i32,
    pub target_radius_px: f64,
    pub px_per_count_y: f64,
    pub feedback_delay_ms: f64,
}

#[derive(Clone, Copy, Debug)]
struct SeenFrame {
    capture_ns: u64,
    error_x_px: f64,
    error_y_px: f64,
}

#[derive(Clone, Copy, Debug)]
struct SentMove {
    visible_after_ns: u64,
    y_counts: i32,
}

#[derive(Debug, Default)]
pub(super) struct FireStabilizer {
    target_id: Option<u64>,
    previous: Option<SeenFrame>,
    pending: VecDeque<SentMove>,
    learned_rate_counts_s: f64,
    fractional_counts: f64,
    positive_residuals: u8,
    last_send_ns: Option<u64>,
}

impl FireStabilizer {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Returns a final Y demand before the existing device-axis clamp.
    pub fn adjust(&mut self, config: FireStabilizationConfig, sample: FireObservation) -> i32 {
        if !config.enabled
            || config.strength <= 0.0
            || !sample.left_held
            || sample.target_reset
            || !sample.error_x_px.is_finite()
            || !sample.error_y_px.is_finite()
            || !sample.px_per_count_y.is_finite()
            || sample.px_per_count_y <= 0.0
            || !sample.feedback_delay_ms.is_finite()
            || sample.feedback_delay_ms < 0.0
            || !sample.target_radius_px.is_finite()
            || sample.target_radius_px <= 0.0
        {
            self.reset();
            return sample.base_y_counts;
        }

        let near_radius = (sample.target_radius_px * 0.16).clamp(8.0, 30.0);
        if sample.error_x_px.hypot(sample.error_y_px) > near_radius {
            self.reset();
            return sample.base_y_counts;
        }
        if self.target_id != Some(sample.target_id) {
            self.reset();
            self.target_id = Some(sample.target_id);
        }

        let mut matured_y = 0_i32;
        let mut matured = false;
        while self
            .pending
            .front()
            .is_some_and(|move_| move_.visible_after_ns <= sample.capture_ns)
        {
            let sent = self.pending.pop_front().expect("front exists");
            matured_y = matured_y.saturating_add(sent.y_counts);
            matured = true;
        }

        if let Some(previous) = self.previous {
            let gap_ns = sample.capture_ns.saturating_sub(previous.capture_ns);
            if sample.capture_ns <= previous.capture_ns || gap_ns > MAX_OBSERVATION_GAP_NS {
                self.reset();
                self.target_id = Some(sample.target_id);
            } else if matured {
                let dt_s = gap_ns as f64 / 1_000_000_000.0;
                let residual_px = sample.error_y_px - previous.error_y_px
                    + f64::from(matured_y) * sample.px_per_count_y;
                let moving_horizontally =
                    (sample.error_x_px - previous.error_x_px).abs() > near_radius * 0.5;
                if !moving_horizontally
                    && residual_px > RESIDUAL_NOISE_PX
                    && sample.error_y_px >= 0.0
                {
                    self.positive_residuals = self.positive_residuals.saturating_add(1);
                    if self.positive_residuals >= 2 {
                        let observed_rate = residual_px / sample.px_per_count_y / dt_s;
                        let cap = 2_400.0 * config.strength;
                        let alpha = (dt_s / 0.08).clamp(0.0, 1.0) * config.strength;
                        self.learned_rate_counts_s +=
                            alpha * (observed_rate.min(cap) - self.learned_rate_counts_s);
                    }
                } else {
                    self.positive_residuals = 0;
                    if residual_px < -RESIDUAL_NOISE_PX || moving_horizontally {
                        self.learned_rate_counts_s *= (1.0 - dt_s / 0.06).clamp(0.0, 1.0);
                    }
                }
            }
        }
        self.previous = Some(SeenFrame {
            capture_ns: sample.capture_ns,
            error_x_px: sample.error_x_px,
            error_y_px: sample.error_y_px,
        });

        if sample.error_y_px <= -OVERSHOOT_PX {
            self.learned_rate_counts_s = 0.0;
            self.fractional_counts = 0.0;
            return sample.base_y_counts;
        }
        if sample.error_y_px < 0.0 {
            self.learned_rate_counts_s *= 0.75;
            self.fractional_counts = 0.0;
            return if self.learned_rate_counts_s > 0.0 {
                (f64::from(sample.base_y_counts) * 0.4).round() as i32
            } else {
                sample.base_y_counts
            };
        }

        let dt_s = self
            .last_send_ns
            .and_then(|last| sample.issued_ns.checked_sub(last))
            .map_or(0.0, |ns| (ns as f64 / 1_000_000_000.0).min(0.05));
        let extra = self.learned_rate_counts_s * dt_s + self.fractional_counts;
        let allowable = (sample.error_y_px / sample.px_per_count_y
            - f64::from(sample.base_y_counts.max(0))
            + 1.0)
            .floor()
            .clamp(0.0, i32::MAX as f64) as i32;
        let extra_counts = (extra.floor() as i32).clamp(0, allowable);
        self.fractional_counts = if extra_counts == allowable {
            0.0
        } else {
            extra.fract()
        };
        sample.base_y_counts.saturating_add(extra_counts)
    }

    pub fn record_sent(&mut self, issued_ns: u64, feedback_delay_ms: f64, y_counts: i32) {
        if self.target_id.is_none() {
            return;
        }
        if self.pending.len() == MAX_PENDING_SENDS {
            self.reset();
            return;
        }
        let delay_ns = (feedback_delay_ms.clamp(0.0, 100.0) * 1_000_000.0) as u64;
        self.pending.push_back(SentMove {
            visible_after_ns: issued_ns.saturating_add(delay_ns),
            y_counts,
        });
        self.last_send_ns = Some(issued_ns);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_only_from_confirmed_near_target_fire_and_resets_on_release() {
        let mut stabilizer = FireStabilizer::default();
        let config = FireStabilizationConfig {
            enabled: true,
            strength: 1.0,
        };
        let mut sample = FireObservation {
            target_id: 1,
            capture_ns: 1_000_000_000,
            issued_ns: 1_001_000_000,
            error_x_px: 0.0,
            error_y_px: 5.0,
            target_reset: false,
            left_held: true,
            base_y_counts: 1,
            target_radius_px: 180.0,
            px_per_count_y: 1.0,
            feedback_delay_ms: 0.0,
        };
        for _ in 0..16 {
            let output = stabilizer.adjust(config, sample);
            stabilizer.record_sent(sample.issued_ns, 0.0, output);
            sample.capture_ns += 20_000_000;
            sample.issued_ns += 20_000_000;
        }
        assert!(stabilizer.learned_rate_counts_s > 0.0);
        assert!(stabilizer.adjust(config, sample) > sample.base_y_counts);

        sample.error_y_px = -5.0;
        assert_eq!(stabilizer.adjust(config, sample), sample.base_y_counts);
        assert_eq!(stabilizer.learned_rate_counts_s, 0.0);

        sample.left_held = false;
        assert_eq!(stabilizer.adjust(config, sample), sample.base_y_counts);
        assert_eq!(stabilizer.learned_rate_counts_s, 0.0);

        sample.left_held = true;
        sample.error_x_px = 80.0;
        assert_eq!(stabilizer.adjust(config, sample), sample.base_y_counts);
    }
}
