//! Core algorithm trace scoring.
//!
//! These metrics intentionally sit next to targeting and control instead of in
//! Studio or scripts. They score whether a sequence of visual observations and
//! emitted counts actually converges, without changing the controller itself.

use serde::{Deserialize, Serialize};

use crate::controller::ControlDecision;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AlgorithmScoreConfig {
    /// Visual error radius considered settled.
    pub settle_radius_px: f64,
    /// Consecutive in-radius target samples required before reporting settle.
    pub settle_window: usize,
    /// Fraction of valid samples used for tail-error averaging.
    pub tail_fraction: f64,
}

impl Default for AlgorithmScoreConfig {
    fn default() -> Self {
        Self {
            settle_radius_px: 1.0,
            settle_window: 3,
            tail_fraction: 0.25,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AlgorithmTraceSample {
    pub generation: u64,
    #[serde(default)]
    pub capture_ts_ns: u64,
    #[serde(default)]
    pub control_now_ns: u64,
    pub target_id: Option<u64>,
    pub target_valid: bool,
    pub observed_error_x_px: f64,
    pub observed_error_y_px: f64,
    pub emitted_counts_x: i32,
    pub emitted_counts_y: i32,
    pub emit_allowed: bool,
    pub frame_age_ms: f64,
}

impl AlgorithmTraceSample {
    pub fn from_control_decision(decision: &ControlDecision) -> Self {
        let target_id =
            (decision.sample_available && decision.target_id != 0).then_some(decision.target_id);
        Self {
            generation: decision.generation,
            capture_ts_ns: decision.capture_ts_ns,
            control_now_ns: decision.control_now_ns,
            target_id,
            target_valid: target_id.is_some(),
            observed_error_x_px: decision.observed_error_x,
            observed_error_y_px: decision.observed_error_y,
            emitted_counts_x: decision.dx,
            emitted_counts_y: decision.dy,
            emit_allowed: decision.emit_allowed,
            frame_age_ms: decision.frame_age_ms,
        }
    }

    pub fn error_radius_px(self) -> f64 {
        self.observed_error_x_px.hypot(self.observed_error_y_px)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CountResponseModel {
    pub px_per_count_x: f64,
    pub px_per_count_y: f64,
    pub min_lag_samples: usize,
    pub max_lag_samples: usize,
}

impl CountResponseModel {
    pub fn is_valid(self) -> bool {
        self.px_per_count_x.is_finite()
            && self.px_per_count_x > 0.0
            && self.px_per_count_y.is_finite()
            && self.px_per_count_y > 0.0
            && self.min_lag_samples > 0
            && self.max_lag_samples >= self.min_lag_samples
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CountResponseLagEstimate {
    pub best: CountResponseLagScore,
    pub candidates: Vec<CountResponseLagScore>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CountResponseLagScore {
    pub lag_samples: usize,
    pub sample_pairs: usize,
    pub mean_delay_ms: f64,
    pub rms_residual_px: f64,
    pub mean_signed_residual_x_px: f64,
    pub mean_signed_residual_y_px: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AlgorithmScore {
    pub total_samples: usize,
    pub valid_target_samples: usize,
    pub emitted_commands: usize,
    pub non_emit_samples: usize,
    pub target_switches: usize,
    pub initial_error_px: f64,
    pub final_error_px: f64,
    pub best_error_px: f64,
    pub mean_error_px: f64,
    pub rms_error_px: f64,
    pub mean_tail_error_px: f64,
    pub max_overshoot_px: f64,
    pub settle_generation: Option<u64>,
    pub sign_flips_x: usize,
    pub sign_flips_y: usize,
    pub mean_frame_age_ms: f64,
}

pub fn score_algorithm_trace(
    samples: &[AlgorithmTraceSample],
    config: AlgorithmScoreConfig,
) -> AlgorithmScore {
    let valid = samples
        .iter()
        .copied()
        .filter(|sample| {
            sample.target_valid
                && sample.observed_error_x_px.is_finite()
                && sample.observed_error_y_px.is_finite()
                && sample.frame_age_ms.is_finite()
        })
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return AlgorithmScore {
            total_samples: samples.len(),
            non_emit_samples: samples.iter().filter(|sample| !sample.emit_allowed).count(),
            ..AlgorithmScore::default()
        };
    }

    let errors = valid
        .iter()
        .map(|sample| sample.error_radius_px())
        .collect::<Vec<_>>();
    let initial = valid[0];
    let final_sample = valid[valid.len() - 1];
    let tail_len = ((valid.len() as f64) * config.tail_fraction.clamp(0.0, 1.0))
        .ceil()
        .max(1.0) as usize;
    let tail_start = valid.len().saturating_sub(tail_len);
    let total_error = errors.iter().sum::<f64>();
    let total_squared_error = errors.iter().map(|error| error * error).sum::<f64>();
    let best_error = errors.iter().copied().fold(f64::INFINITY, f64::min);
    let mean_frame_age_ms =
        valid.iter().map(|sample| sample.frame_age_ms).sum::<f64>() / valid.len() as f64;

    AlgorithmScore {
        total_samples: samples.len(),
        valid_target_samples: valid.len(),
        emitted_commands: samples.iter().filter(|sample| sample.emit_allowed).count(),
        non_emit_samples: samples.iter().filter(|sample| !sample.emit_allowed).count(),
        target_switches: count_target_switches(&valid),
        initial_error_px: initial.error_radius_px(),
        final_error_px: final_sample.error_radius_px(),
        best_error_px: best_error,
        mean_error_px: total_error / valid.len() as f64,
        rms_error_px: (total_squared_error / valid.len() as f64).sqrt(),
        mean_tail_error_px: errors[tail_start..].iter().sum::<f64>() / tail_len as f64,
        max_overshoot_px: max_overshoot(&valid),
        settle_generation: settle_generation(&valid, config),
        sign_flips_x: count_sign_flips(&valid, Axis::X),
        sign_flips_y: count_sign_flips(&valid, Axis::Y),
        mean_frame_age_ms,
    }
}

/// Estimate which delayed count-response window best explains visual error
/// deltas. This assumes a mostly stationary target/calibration trace; target
/// motion remains in the residual and is the signal that self-motion
/// subtraction still needs better capture data.
pub fn estimate_count_response_lag(
    samples: &[AlgorithmTraceSample],
    model: CountResponseModel,
) -> Option<CountResponseLagEstimate> {
    if !model.is_valid() || samples.len() < model.min_lag_samples.saturating_add(2) {
        return None;
    }
    let candidates = (model.min_lag_samples..=model.max_lag_samples)
        .filter_map(|lag| score_count_response_lag(samples, model, lag))
        .collect::<Vec<_>>();
    let best = candidates.iter().copied().min_by(|left, right| {
        left.rms_residual_px
            .total_cmp(&right.rms_residual_px)
            .then_with(|| right.sample_pairs.cmp(&left.sample_pairs))
            .then_with(|| left.lag_samples.cmp(&right.lag_samples))
    })?;
    Some(CountResponseLagEstimate { best, candidates })
}

fn score_count_response_lag(
    samples: &[AlgorithmTraceSample],
    model: CountResponseModel,
    lag: usize,
) -> Option<CountResponseLagScore> {
    let mut pairs = 0;
    let mut squared_residual_sum = 0.0;
    let mut residual_x_sum = 0.0;
    let mut residual_y_sum = 0.0;
    let mut delay_ms_sum = 0.0;
    let mut delay_pairs = 0;
    for index in lag..samples.len() {
        let command_sample = samples[index - lag];
        if command_sample.emitted_counts_x == 0 && command_sample.emitted_counts_y == 0 {
            continue;
        }
        let previous = samples[index - 1];
        let current = samples[index];
        if !command_sample.target_valid
            || !previous.target_valid
            || !current.target_valid
            || command_sample.target_id != current.target_id
            || previous.target_id != current.target_id
            || !samples_finite(command_sample)
            || !samples_finite(previous)
            || !samples_finite(current)
        {
            continue;
        }
        let observed_delta_x = current.observed_error_x_px - previous.observed_error_x_px;
        let observed_delta_y = current.observed_error_y_px - previous.observed_error_y_px;
        let predicted_delta_x = -f64::from(command_sample.emitted_counts_x) * model.px_per_count_x;
        let predicted_delta_y = -f64::from(command_sample.emitted_counts_y) * model.px_per_count_y;
        let residual_x = observed_delta_x - predicted_delta_x;
        let residual_y = observed_delta_y - predicted_delta_y;
        residual_x_sum += residual_x;
        residual_y_sum += residual_y;
        squared_residual_sum += residual_x.mul_add(residual_x, residual_y * residual_y);
        if command_sample.capture_ts_ns != 0
            && current.capture_ts_ns >= command_sample.capture_ts_ns
        {
            delay_ms_sum +=
                (current.capture_ts_ns - command_sample.capture_ts_ns) as f64 / 1_000_000.0;
            delay_pairs += 1;
        }
        pairs += 1;
    }
    (pairs > 0).then(|| CountResponseLagScore {
        lag_samples: lag,
        sample_pairs: pairs,
        mean_delay_ms: if delay_pairs > 0 {
            delay_ms_sum / delay_pairs as f64
        } else {
            0.0
        },
        rms_residual_px: (squared_residual_sum / pairs as f64).sqrt(),
        mean_signed_residual_x_px: residual_x_sum / pairs as f64,
        mean_signed_residual_y_px: residual_y_sum / pairs as f64,
    })
}

fn samples_finite(sample: AlgorithmTraceSample) -> bool {
    sample.observed_error_x_px.is_finite()
        && sample.observed_error_y_px.is_finite()
        && sample.frame_age_ms.is_finite()
}

fn count_target_switches(samples: &[AlgorithmTraceSample]) -> usize {
    let mut previous = None;
    let mut switches = 0;
    for sample in samples {
        let Some(target_id) = sample.target_id else {
            continue;
        };
        if let Some(previous_id) = previous
            && previous_id != target_id
        {
            switches += 1;
        }
        previous = Some(target_id);
    }
    switches
}

fn settle_generation(
    samples: &[AlgorithmTraceSample],
    config: AlgorithmScoreConfig,
) -> Option<u64> {
    let settle_radius = config.settle_radius_px.max(0.0);
    let window = config.settle_window.max(1);
    let mut run = 0;
    for sample in samples {
        if sample.error_radius_px() <= settle_radius {
            run += 1;
            if run >= window {
                return Some(sample.generation);
            }
        } else {
            run = 0;
        }
    }
    None
}

fn max_overshoot(samples: &[AlgorithmTraceSample]) -> f64 {
    let first = samples[0];
    samples
        .iter()
        .map(|sample| {
            let x = overshoot_component(first.observed_error_x_px, sample.observed_error_x_px);
            let y = overshoot_component(first.observed_error_y_px, sample.observed_error_y_px);
            x.hypot(y)
        })
        .fold(0.0, f64::max)
}

fn overshoot_component(initial: f64, current: f64) -> f64 {
    if initial.abs() <= f64::EPSILON || initial.signum() == current.signum() {
        0.0
    } else {
        current.abs()
    }
}

#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
}

fn count_sign_flips(samples: &[AlgorithmTraceSample], axis: Axis) -> usize {
    let mut previous = 0;
    let mut flips = 0;
    for sample in samples {
        let value = match axis {
            Axis::X => sample.observed_error_x_px,
            Axis::Y => sample.observed_error_y_px,
        };
        let current = finite_sign(value);
        if current == 0 {
            continue;
        }
        if previous != 0 && current != previous {
            flips += 1;
        }
        previous = current;
    }
    flips
}

fn finite_sign(value: f64) -> i8 {
    if !value.is_finite() || value.abs() <= f64::EPSILON {
        0
    } else if value.is_sign_positive() {
        1
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AlgorithmScoreConfig, AlgorithmTraceSample, CountResponseModel,
        estimate_count_response_lag, score_algorithm_trace,
    };

    #[test]
    fn score_reports_convergence_and_tail_error() {
        let samples = [
            sample(1, 10.0, 0.0, 4, true),
            sample(2, 4.0, 0.0, 2, true),
            sample(3, 0.8, 0.0, 1, true),
            sample(4, 0.4, 0.0, 0, false),
            sample(5, 0.2, 0.0, 0, false),
        ];

        let score = score_algorithm_trace(&samples, AlgorithmScoreConfig::default());

        assert_eq!(score.total_samples, 5);
        assert_eq!(score.valid_target_samples, 5);
        assert_eq!(score.emitted_commands, 3);
        assert_eq!(score.non_emit_samples, 2);
        assert_eq!(score.initial_error_px, 10.0);
        assert_eq!(score.final_error_px, 0.2);
        assert_eq!(score.best_error_px, 0.2);
        assert_eq!(score.settle_generation, Some(5));
        assert!(score.mean_tail_error_px < 1.0);
    }

    #[test]
    fn score_counts_switches_flips_and_overshoot() {
        let samples = [
            sample(1, 6.0, 2.0, 3, true),
            AlgorithmTraceSample {
                target_id: Some(2),
                ..sample(2, -1.5, -0.5, -1, true)
            },
        ];

        let score = score_algorithm_trace(&samples, AlgorithmScoreConfig::default());

        assert_eq!(score.target_switches, 1);
        assert_eq!(score.sign_flips_x, 1);
        assert_eq!(score.sign_flips_y, 1);
        assert!(score.max_overshoot_px > 1.5);
    }

    #[test]
    fn count_response_lag_estimator_finds_the_delayed_visual_effect() {
        let counts = [0, 4, 2, 0, -1, 0, 0, 0];
        let true_lag = 2;
        let px_per_count = 1.5;
        let mut error_x = 20.0;
        let mut samples = Vec::new();
        for index in 0..counts.len() {
            if index >= true_lag {
                error_x -= f64::from(counts[index - true_lag]) * px_per_count;
            }
            samples.push(AlgorithmTraceSample {
                generation: index as u64 + 1,
                capture_ts_ns: 1_000_000_000 + index as u64 * 10_000_000,
                control_now_ns: 1_004_000_000 + index as u64 * 10_000_000,
                target_id: Some(1),
                target_valid: true,
                observed_error_x_px: error_x,
                observed_error_y_px: 0.0,
                emitted_counts_x: counts[index],
                emitted_counts_y: 0,
                emit_allowed: counts[index] != 0,
                frame_age_ms: 4.0,
            });
        }

        let estimate = estimate_count_response_lag(
            &samples,
            CountResponseModel {
                px_per_count_x: px_per_count,
                px_per_count_y: px_per_count,
                min_lag_samples: 1,
                max_lag_samples: 4,
            },
        )
        .expect("estimate");

        assert_eq!(estimate.best.lag_samples, true_lag);
        assert_eq!(estimate.best.sample_pairs, 3);
        assert!((estimate.best.mean_delay_ms - 20.0).abs() < 1e-12);
        assert!(estimate.best.rms_residual_px < 1e-12);
    }

    fn sample(
        generation: u64,
        error_x: f64,
        error_y: f64,
        counts_x: i32,
        emit_allowed: bool,
    ) -> AlgorithmTraceSample {
        AlgorithmTraceSample {
            generation,
            capture_ts_ns: 1_000_000_000 + generation * 10_000_000,
            control_now_ns: 1_004_000_000 + generation * 10_000_000,
            target_id: Some(1),
            target_valid: true,
            observed_error_x_px: error_x,
            observed_error_y_px: error_y,
            emitted_counts_x: counts_x,
            emitted_counts_y: 0,
            emit_allowed,
            frame_age_ms: 8.0,
        }
    }
}
