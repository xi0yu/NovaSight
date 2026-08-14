//! Core algorithm trace scoring.
//!
//! These metrics intentionally sit next to targeting and control instead of in
//! Studio or scripts. They score whether a sequence of visual observations and
//! emitted counts actually converges, without changing the controller itself.

use serde::{Deserialize, Serialize};

use crate::controller::AimResult;
use crate::prediction::PredictionMotionState;

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
    pub fn from_aim_result(result: &AimResult) -> Self {
        let decision = result;
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionTruthProjection {
    Raw,
    ConfidenceWeighted,
    #[default]
    Capped,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictionTruthConfig {
    pub horizons_ms: Vec<f64>,
    pub projection: PredictionTruthProjection,
    pub static_speed_px_ms: f64,
    pub acceleration_deadband_px_ms2: f64,
    pub jitter_trend_threshold: f64,
}

impl Default for PredictionTruthConfig {
    fn default() -> Self {
        Self {
            horizons_ms: vec![5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            projection: PredictionTruthProjection::Capped,
            static_speed_px_ms: 0.02,
            acceleration_deadband_px_ms2: 0.005,
            jitter_trend_threshold: 0.25,
        }
    }
}

impl PredictionTruthConfig {
    fn valid_horizons(&self) -> Vec<f64> {
        let mut horizons = self
            .horizons_ms
            .iter()
            .copied()
            .filter(|horizon| horizon.is_finite() && *horizon > 0.0)
            .collect::<Vec<_>>();
        horizons.sort_by(f64::total_cmp);
        horizons.dedup_by(|left, right| (*left - *right).abs() < 1e-9);
        horizons
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PredictionTruthSample {
    pub generation: u64,
    pub capture_ts_ns: u64,
    pub target_id: Option<u64>,
    pub target_valid: bool,
    pub aim_x_px: f64,
    pub aim_y_px: f64,
    pub velocity_x_px_ms: f64,
    pub velocity_y_px_ms: f64,
    pub motion_state_x: PredictionMotionState,
    pub motion_state_y: PredictionMotionState,
    pub trend_consistency_x: f64,
    pub trend_consistency_y: f64,
    pub acceleration_x_px_ms2: f64,
    pub acceleration_y_px_ms2: f64,
    pub motion_confidence_x: f64,
    pub motion_confidence_y: f64,
    pub prediction_cap_x_px: f64,
    pub prediction_cap_y_px: f64,
    pub prediction_allowed_x: bool,
    pub prediction_allowed_y: bool,
}

impl PredictionTruthSample {
    pub fn from_aim_result(result: &AimResult) -> Self {
        let decision = result;
        let target_id =
            (decision.sample_available && decision.target_id != 0).then_some(decision.target_id);
        Self {
            generation: decision.generation,
            capture_ts_ns: decision.capture_ts_ns,
            target_id,
            target_valid: target_id.is_some(),
            aim_x_px: decision.aim_x,
            aim_y_px: decision.aim_y,
            velocity_x_px_ms: decision.velocity_x,
            velocity_y_px_ms: decision.velocity_y,
            motion_state_x: decision.motion_state,
            motion_state_y: decision.motion_state_y,
            trend_consistency_x: 1.0,
            trend_consistency_y: 1.0,
            acceleration_x_px_ms2: 0.0,
            acceleration_y_px_ms2: 0.0,
            motion_confidence_x: 1.0,
            motion_confidence_y: 1.0,
            prediction_cap_x_px: decision.prediction_allowed_cap_x,
            prediction_cap_y_px: decision.prediction_allowed_cap_y,
            prediction_allowed_x: decision.prediction_allowed,
            prediction_allowed_y: decision.prediction_allowed_y,
        }
    }

    fn position_is_usable(self) -> bool {
        self.target_valid
            && self.target_id.is_some()
            && self.capture_ts_ns != 0
            && self.aim_x_px.is_finite()
            && self.aim_y_px.is_finite()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionTruthMotionClass {
    Static,
    Continuous,
    Accelerating,
    Decelerating,
    Jitter,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PredictionTruthHorizonScore {
    pub horizon_ms: f64,
    pub sample_pairs: usize,
    pub mae_px: f64,
    pub rmse_px: f64,
    pub bias_x_px: f64,
    pub bias_y_px: f64,
    pub p90_error_px: f64,
    pub p95_error_px: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PredictionTruthMotionClassScore {
    pub motion_class: PredictionTruthMotionClass,
    pub horizons: Vec<PredictionTruthHorizonScore>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PredictionTruthReport {
    pub total_samples: usize,
    pub valid_position_samples: usize,
    pub projection: PredictionTruthProjection,
    pub horizons: Vec<PredictionTruthHorizonScore>,
    pub motion_classes: Vec<PredictionTruthMotionClassScore>,
}

pub fn score_prediction_truth(
    samples: &[PredictionTruthSample],
    config: PredictionTruthConfig,
) -> PredictionTruthReport {
    let horizons = config.valid_horizons();
    let valid_position_samples = samples
        .iter()
        .filter(|sample| sample.position_is_usable())
        .count();
    let horizon_scores = horizons
        .iter()
        .copied()
        .map(|horizon_ms| score_prediction_horizon(samples, &config, horizon_ms, None))
        .collect::<Vec<_>>();
    let motion_classes = prediction_truth_motion_class_order()
        .into_iter()
        .filter_map(|motion_class| {
            let class_horizons = horizons
                .iter()
                .copied()
                .map(|horizon_ms| {
                    score_prediction_horizon(samples, &config, horizon_ms, Some(motion_class))
                })
                .filter(|score| score.sample_pairs > 0)
                .collect::<Vec<_>>();
            (!class_horizons.is_empty()).then_some(PredictionTruthMotionClassScore {
                motion_class,
                horizons: class_horizons,
            })
        })
        .collect::<Vec<_>>();

    PredictionTruthReport {
        total_samples: samples.len(),
        valid_position_samples,
        projection: config.projection,
        horizons: horizon_scores,
        motion_classes,
    }
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

#[derive(Clone, Copy, Debug, PartialEq)]
struct PredictionTruthResidual {
    error_x_px: f64,
    error_y_px: f64,
    error_radius_px: f64,
}

fn score_prediction_horizon(
    samples: &[PredictionTruthSample],
    config: &PredictionTruthConfig,
    horizon_ms: f64,
    motion_class: Option<PredictionTruthMotionClass>,
) -> PredictionTruthHorizonScore {
    let mut residuals = Vec::new();
    for (index, sample) in samples.iter().copied().enumerate() {
        if motion_class
            .is_some_and(|class| classify_prediction_truth_motion(sample, config) != class)
        {
            continue;
        }
        let Some(future_position) = future_prediction_truth_position(samples, index, horizon_ms)
        else {
            continue;
        };
        let Some((offset_x, offset_y)) = prediction_truth_offset(sample, horizon_ms, config) else {
            continue;
        };
        let predicted_x = sample.aim_x_px + offset_x;
        let predicted_y = sample.aim_y_px + offset_y;
        let error_x = predicted_x - future_position.0;
        let error_y = predicted_y - future_position.1;
        if !error_x.is_finite() || !error_y.is_finite() {
            continue;
        }
        residuals.push(PredictionTruthResidual {
            error_x_px: error_x,
            error_y_px: error_y,
            error_radius_px: error_x.hypot(error_y),
        });
    }
    summarize_prediction_residuals(horizon_ms, &residuals)
}

fn summarize_prediction_residuals(
    horizon_ms: f64,
    residuals: &[PredictionTruthResidual],
) -> PredictionTruthHorizonScore {
    if residuals.is_empty() {
        return PredictionTruthHorizonScore {
            horizon_ms,
            ..PredictionTruthHorizonScore::default()
        };
    }
    let sample_pairs = residuals.len();
    let mae_px = residuals
        .iter()
        .map(|residual| residual.error_radius_px)
        .sum::<f64>()
        / sample_pairs as f64;
    let rmse_px = (residuals
        .iter()
        .map(|residual| {
            residual
                .error_radius_px
                .mul_add(residual.error_radius_px, 0.0)
        })
        .sum::<f64>()
        / sample_pairs as f64)
        .sqrt();
    let bias_x_px = residuals
        .iter()
        .map(|residual| residual.error_x_px)
        .sum::<f64>()
        / sample_pairs as f64;
    let bias_y_px = residuals
        .iter()
        .map(|residual| residual.error_y_px)
        .sum::<f64>()
        / sample_pairs as f64;
    let mut sorted_errors = residuals
        .iter()
        .map(|residual| residual.error_radius_px)
        .collect::<Vec<_>>();
    sorted_errors.sort_by(f64::total_cmp);

    PredictionTruthHorizonScore {
        horizon_ms,
        sample_pairs,
        mae_px,
        rmse_px,
        bias_x_px,
        bias_y_px,
        p90_error_px: percentile_from_sorted(&sorted_errors, 0.90),
        p95_error_px: percentile_from_sorted(&sorted_errors, 0.95),
    }
}

fn prediction_truth_offset(
    sample: PredictionTruthSample,
    horizon_ms: f64,
    config: &PredictionTruthConfig,
) -> Option<(f64, f64)> {
    if !sample.prediction_allowed_x
        || !sample.prediction_allowed_y
        || !sample.velocity_x_px_ms.is_finite()
        || !sample.velocity_y_px_ms.is_finite()
        || !horizon_ms.is_finite()
        || horizon_ms <= 0.0
    {
        return None;
    }
    let raw_x = sample.velocity_x_px_ms * horizon_ms;
    let raw_y = sample.velocity_y_px_ms * horizon_ms;
    if !raw_x.is_finite() || !raw_y.is_finite() {
        return None;
    }

    match config.projection {
        PredictionTruthProjection::Raw => Some((raw_x, raw_y)),
        PredictionTruthProjection::ConfidenceWeighted => {
            let confidence = prediction_truth_confidence(sample)?;
            let weighted_x = raw_x * confidence;
            let weighted_y = raw_y * confidence;
            (weighted_x.is_finite() && weighted_y.is_finite()).then_some((weighted_x, weighted_y))
        }
        PredictionTruthProjection::Capped => {
            let confidence = prediction_truth_confidence(sample)?;
            let cap_px = prediction_truth_vector_cap(sample)?;
            let weighted_x = raw_x * confidence;
            let weighted_y = raw_y * confidence;
            clamp_prediction_vector(weighted_x, weighted_y, cap_px)
        }
    }
}

fn prediction_truth_confidence(sample: PredictionTruthSample) -> Option<f64> {
    if !sample.motion_confidence_x.is_finite() || !sample.motion_confidence_y.is_finite() {
        return None;
    }
    Some(
        sample
            .motion_confidence_x
            .min(sample.motion_confidence_y)
            .clamp(0.0, 1.0),
    )
}

fn prediction_truth_vector_cap(sample: PredictionTruthSample) -> Option<f64> {
    if !sample.prediction_cap_x_px.is_finite()
        || !sample.prediction_cap_y_px.is_finite()
        || sample.prediction_cap_x_px < 0.0
        || sample.prediction_cap_y_px < 0.0
    {
        return None;
    }
    Some(sample.prediction_cap_x_px.min(sample.prediction_cap_y_px))
}

fn clamp_prediction_vector(x: f64, y: f64, cap_px: f64) -> Option<(f64, f64)> {
    if !x.is_finite() || !y.is_finite() || !cap_px.is_finite() || cap_px < 0.0 {
        return None;
    }
    let magnitude = x.hypot(y);
    if magnitude <= cap_px || magnitude <= f64::EPSILON {
        return Some((x, y));
    }
    let scale = cap_px / magnitude;
    Some((x * scale, y * scale))
}

fn future_prediction_truth_position(
    samples: &[PredictionTruthSample],
    index: usize,
    horizon_ms: f64,
) -> Option<(f64, f64)> {
    let sample = samples.get(index).copied()?;
    if !sample.position_is_usable() || !horizon_ms.is_finite() || horizon_ms <= 0.0 {
        return None;
    }
    let target_id = sample.target_id?;
    let future_ts_ns = sample
        .capture_ts_ns
        .checked_add((horizon_ms * 1_000_000.0).round() as u64)?;
    let mut left = sample;
    for right in samples.iter().copied().skip(index + 1) {
        if !right.position_is_usable() {
            continue;
        }
        if right.target_id != Some(target_id) {
            return None;
        }
        if right.capture_ts_ns <= left.capture_ts_ns {
            continue;
        }
        if right.capture_ts_ns >= future_ts_ns {
            return Some(interpolate_prediction_truth_position(
                left,
                right,
                future_ts_ns,
            ));
        }
        left = right;
    }
    None
}

fn interpolate_prediction_truth_position(
    left: PredictionTruthSample,
    right: PredictionTruthSample,
    target_ts_ns: u64,
) -> (f64, f64) {
    if target_ts_ns <= left.capture_ts_ns {
        return (left.aim_x_px, left.aim_y_px);
    }
    if target_ts_ns >= right.capture_ts_ns {
        return (right.aim_x_px, right.aim_y_px);
    }
    let span = (right.capture_ts_ns - left.capture_ts_ns) as f64;
    let weight = (target_ts_ns - left.capture_ts_ns) as f64 / span;
    (
        left.aim_x_px + (right.aim_x_px - left.aim_x_px) * weight,
        left.aim_y_px + (right.aim_y_px - left.aim_y_px) * weight,
    )
}

fn classify_prediction_truth_motion(
    sample: PredictionTruthSample,
    config: &PredictionTruthConfig,
) -> PredictionTruthMotionClass {
    let velocity_x = sample.velocity_x_px_ms;
    let velocity_y = sample.velocity_y_px_ms;
    let acceleration_x = sample.acceleration_x_px_ms2;
    let acceleration_y = sample.acceleration_y_px_ms2;
    if !velocity_x.is_finite()
        || !velocity_y.is_finite()
        || !acceleration_x.is_finite()
        || !acceleration_y.is_finite()
    {
        return PredictionTruthMotionClass::Unknown;
    }
    let speed = velocity_x.hypot(velocity_y);
    if matches!(
        (sample.motion_state_x, sample.motion_state_y),
        (
            PredictionMotionState::Stationary,
            PredictionMotionState::Stationary
        )
    ) || speed <= config.static_speed_px_ms.max(0.0)
    {
        return PredictionTruthMotionClass::Static;
    }
    let trend = sample
        .trend_consistency_x
        .min(sample.trend_consistency_y)
        .clamp(0.0, 1.0);
    if trend < config.jitter_trend_threshold.clamp(0.0, 1.0) {
        return PredictionTruthMotionClass::Jitter;
    }
    let acceleration_along_velocity =
        velocity_x.mul_add(acceleration_x, velocity_y * acceleration_y) / speed.max(1e-9);
    let acceleration_deadband = config.acceleration_deadband_px_ms2.max(0.0);
    if acceleration_along_velocity > acceleration_deadband {
        PredictionTruthMotionClass::Accelerating
    } else if acceleration_along_velocity < -acceleration_deadband {
        PredictionTruthMotionClass::Decelerating
    } else {
        PredictionTruthMotionClass::Continuous
    }
}

fn prediction_truth_motion_class_order() -> [PredictionTruthMotionClass; 6] {
    [
        PredictionTruthMotionClass::Static,
        PredictionTruthMotionClass::Continuous,
        PredictionTruthMotionClass::Accelerating,
        PredictionTruthMotionClass::Decelerating,
        PredictionTruthMotionClass::Jitter,
        PredictionTruthMotionClass::Unknown,
    ]
}

fn percentile_from_sorted(sorted_values: &[f64], percentile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let percentile = percentile.clamp(0.0, 1.0);
    let index = ((sorted_values.len() - 1) as f64 * percentile).ceil() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
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
        AlgorithmScoreConfig, AlgorithmTraceSample, CountResponseModel, PredictionTruthConfig,
        PredictionTruthMotionClass, PredictionTruthProjection, PredictionTruthSample,
        estimate_count_response_lag, score_algorithm_trace, score_prediction_truth,
    };
    use crate::prediction::PredictionMotionState;

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

    #[test]
    fn prediction_truth_scores_multiple_horizons_against_future_position() {
        let samples = (0..7)
            .map(|index| {
                prediction_truth_sample(
                    index + 1,
                    100.0 + index as f64 * 10.0,
                    80.0,
                    1.0,
                    0.0,
                    PredictionMotionState::Continuous,
                    PredictionMotionState::Stationary,
                )
            })
            .collect::<Vec<_>>();

        let report = score_prediction_truth(
            &samples,
            PredictionTruthConfig {
                horizons_ms: vec![20.0, 10.0, 10.0],
                projection: PredictionTruthProjection::Capped,
                ..PredictionTruthConfig::default()
            },
        );

        assert_eq!(report.total_samples, 7);
        assert_eq!(report.valid_position_samples, 7);
        assert_eq!(report.horizons.len(), 2);
        assert_eq!(report.horizons[0].horizon_ms, 10.0);
        assert_eq!(report.horizons[0].sample_pairs, 6);
        assert!(report.horizons[0].mae_px < 1e-12);
        assert_eq!(report.horizons[1].horizon_ms, 20.0);
        assert_eq!(report.horizons[1].sample_pairs, 5);
        assert!(report.horizons[1].rmse_px < 1e-12);
        assert!(
            report
                .motion_classes
                .iter()
                .any(|score| score.motion_class == PredictionTruthMotionClass::Continuous)
        );
    }

    #[test]
    fn prediction_truth_projection_separates_raw_confidence_and_vector_cap_error() {
        let mut samples = (0..5)
            .map(|index| {
                prediction_truth_sample(
                    index + 1,
                    100.0 + index as f64 * 10.0,
                    80.0,
                    1.0,
                    0.0,
                    PredictionMotionState::Continuous,
                    PredictionMotionState::Stationary,
                )
            })
            .collect::<Vec<_>>();
        for sample in &mut samples {
            sample.motion_confidence_x = 0.2;
            sample.prediction_cap_x_px = 2.0;
        }
        let base = PredictionTruthConfig {
            horizons_ms: vec![10.0],
            ..PredictionTruthConfig::default()
        };

        let raw = score_prediction_truth(
            &samples,
            PredictionTruthConfig {
                projection: PredictionTruthProjection::Raw,
                ..base.clone()
            },
        );
        let confidence_weighted = score_prediction_truth(
            &samples,
            PredictionTruthConfig {
                projection: PredictionTruthProjection::ConfidenceWeighted,
                ..base.clone()
            },
        );
        let capped = score_prediction_truth(
            &samples,
            PredictionTruthConfig {
                projection: PredictionTruthProjection::Capped,
                ..base
            },
        );

        assert!(raw.horizons[0].mae_px < 1e-12);
        assert!((confidence_weighted.horizons[0].mae_px - 8.0).abs() < 1e-12);
        assert!((confidence_weighted.horizons[0].bias_x_px + 8.0).abs() < 1e-12);
        assert!((capped.horizons[0].mae_px - 8.0).abs() < 1e-12);
        assert!((capped.horizons[0].bias_x_px + 8.0).abs() < 1e-12);
    }

    #[test]
    fn prediction_truth_capped_projection_uses_vector_cap() {
        let mut samples = (0..5)
            .map(|index| {
                prediction_truth_sample(
                    index + 1,
                    100.0 + index as f64 * 10.0,
                    80.0 + index as f64 * 5.0,
                    1.0,
                    0.5,
                    PredictionMotionState::Continuous,
                    PredictionMotionState::Continuous,
                )
            })
            .collect::<Vec<_>>();
        for sample in &mut samples {
            sample.prediction_cap_x_px = 5.0;
            sample.prediction_cap_y_px = 5.0;
        }

        let capped = score_prediction_truth(
            &samples,
            PredictionTruthConfig {
                horizons_ms: vec![10.0],
                projection: PredictionTruthProjection::Capped,
                ..PredictionTruthConfig::default()
            },
        );

        assert_eq!(capped.horizons[0].sample_pairs, 4);
        assert!((capped.horizons[0].mae_px - (10.0_f64.hypot(5.0) - 5.0)).abs() < 1e-12);
        assert!(capped.horizons[0].bias_x_px < -5.0);
        assert!(capped.horizons[0].bias_y_px < -2.0);
    }

    #[test]
    fn prediction_truth_raw_projection_uses_velocity_time_lead_only() {
        let mut continuous = (0..5)
            .map(|index| {
                prediction_truth_sample(
                    index + 1,
                    100.0 + index as f64 * 11.0,
                    80.0,
                    1.0,
                    0.0,
                    PredictionMotionState::Continuous,
                    PredictionMotionState::Stationary,
                )
            })
            .collect::<Vec<_>>();
        for sample in &mut continuous {
            sample.acceleration_x_px_ms2 = 0.1;
        }
        let mut unstable = continuous.clone();
        for sample in &mut unstable {
            sample.motion_state_x = PredictionMotionState::Unstable;
        }

        let config = PredictionTruthConfig {
            horizons_ms: vec![10.0],
            projection: PredictionTruthProjection::Raw,
            ..PredictionTruthConfig::default()
        };
        let continuous_report = score_prediction_truth(&continuous, config.clone());
        let unstable_report = score_prediction_truth(&unstable, config);

        assert!((continuous_report.horizons[0].mae_px - 1.0).abs() < 1e-12);
        assert!((unstable_report.horizons[0].mae_px - 1.0).abs() < 1e-12);
    }

    #[test]
    fn prediction_truth_groups_low_trend_motion_as_jitter() {
        let mut samples = (0..4)
            .map(|index| {
                prediction_truth_sample(
                    index + 1,
                    100.0 + index as f64 * 4.0,
                    80.0,
                    0.4,
                    0.0,
                    PredictionMotionState::Unstable,
                    PredictionMotionState::Stationary,
                )
            })
            .collect::<Vec<_>>();
        for sample in &mut samples {
            sample.trend_consistency_x = 0.10;
        }

        let report = score_prediction_truth(
            &samples,
            PredictionTruthConfig {
                horizons_ms: vec![10.0],
                projection: PredictionTruthProjection::Raw,
                ..PredictionTruthConfig::default()
            },
        );

        let jitter = report
            .motion_classes
            .iter()
            .find(|score| score.motion_class == PredictionTruthMotionClass::Jitter)
            .expect("jitter bucket");
        assert_eq!(jitter.horizons[0].sample_pairs, 3);
        assert!(jitter.horizons[0].mae_px < 1e-12);
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

    fn prediction_truth_sample(
        generation: u64,
        aim_x_px: f64,
        aim_y_px: f64,
        velocity_x_px_ms: f64,
        velocity_y_px_ms: f64,
        motion_state_x: PredictionMotionState,
        motion_state_y: PredictionMotionState,
    ) -> PredictionTruthSample {
        PredictionTruthSample {
            generation,
            capture_ts_ns: 1_000_000_000 + (generation - 1) * 10_000_000,
            target_id: Some(1),
            target_valid: true,
            aim_x_px,
            aim_y_px,
            velocity_x_px_ms,
            velocity_y_px_ms,
            motion_state_x,
            motion_state_y,
            trend_consistency_x: 1.0,
            trend_consistency_y: 1.0,
            acceleration_x_px_ms2: 0.0,
            acceleration_y_px_ms2: 0.0,
            motion_confidence_x: 1.0,
            motion_confidence_y: 1.0,
            prediction_cap_x_px: 1_000.0,
            prediction_cap_y_px: 1_000.0,
            prediction_allowed_x: true,
            prediction_allowed_y: true,
        }
    }
}
