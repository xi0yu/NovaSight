//! Optional target-relative output shaping profiles and telemetry.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

const MINIMUM_JERK_POINTS: usize = 9;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionTiming {
    #[serde(default = "default_timing_model")]
    pub model: String,
    #[serde(default = "default_fitts_a_ms")]
    pub fitts_a_ms: f64,
    #[serde(default = "default_fitts_b_ms")]
    pub fitts_b_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SideCurve {
    #[serde(default = "default_side_model")]
    pub model: String,
    pub control_points: [f64; 4],
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub samples: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DistanceProfile {
    pub sample_count: usize,
    pub progress_curve: Vec<f64>,
    pub side_offset_curve: SideCurve,
    pub median_duration_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionRuntimeParameters {
    #[serde(default = "default_correction_start_ratio")]
    pub correction_start_ratio: f64,
    #[serde(default = "default_correction_gain")]
    pub correction_gain: f64,
    #[serde(default = "default_true")]
    pub spatial_curve_enabled: bool,
    #[serde(default = "default_one")]
    pub side_scale: f64,
    #[serde(default = "default_max_side_ratio")]
    pub max_side_ratio: f64,
    #[serde(default = "default_near_fade_start_px")]
    pub near_fade_start_px: f64,
    #[serde(default = "default_micro_bypass_px")]
    pub micro_bypass_px: f64,
    #[serde(default = "default_dynamic_rebase_ratio")]
    pub dynamic_rebase_ratio: f64,
    #[serde(default = "default_true")]
    pub minimum_jerk_fallback: bool,
    #[serde(default = "default_terminal_feedback_gain")]
    pub terminal_feedback_gain: f64,
}

impl Default for MotionRuntimeParameters {
    fn default() -> Self {
        Self {
            correction_start_ratio: default_correction_start_ratio(),
            correction_gain: default_correction_gain(),
            spatial_curve_enabled: true,
            side_scale: 1.0,
            max_side_ratio: default_max_side_ratio(),
            near_fade_start_px: default_near_fade_start_px(),
            micro_bypass_px: default_micro_bypass_px(),
            dynamic_rebase_ratio: default_dynamic_rebase_ratio(),
            minimum_jerk_fallback: true,
            terminal_feedback_gain: default_terminal_feedback_gain(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionProfile {
    pub profile_version: u32,
    pub profile_id: String,
    pub name: String,
    #[serde(default)]
    pub session_id: String,
    pub sample_count: usize,
    #[serde(default)]
    pub quality_score: u32,
    #[serde(default)]
    pub features: BTreeMap<String, f64>,
    pub timing: MotionTiming,
    #[serde(default)]
    pub progress_curve: Vec<f64>,
    pub side_offset_curve: SideCurve,
    #[serde(default)]
    pub distance_profiles: BTreeMap<String, DistanceProfile>,
    #[serde(default)]
    pub direction_duration_scales: BTreeMap<String, f64>,
    #[serde(default)]
    pub runtime_parameters: MotionRuntimeParameters,
    #[serde(default)]
    pub profile_source: String,
}

impl MotionProfile {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.profile_version != 4 {
            return Err("motion profile_version must be 4");
        }
        if self.profile_id.is_empty() || self.profile_id.len() > 128 {
            return Err("motion profile_id must contain 1..=128 characters");
        }
        if self.name.trim().is_empty() || self.name.len() > 256 {
            return Err("motion profile name must contain 1..=256 characters");
        }
        if self.timing.model != "fitts"
            || !finite_in(self.timing.fitts_a_ms, 0.0, 1_000.0)
            || !finite_in(self.timing.fitts_b_ms, 1.0, 1_000.0)
        {
            return Err("motion profile timing is invalid");
        }
        validate_curve(&self.progress_curve)?;
        validate_side_curve(&self.side_offset_curve)?;
        for profile in self.distance_profiles.values() {
            validate_curve(&profile.progress_curve)?;
            validate_side_curve(&profile.side_offset_curve)?;
            if !finite_in(profile.median_duration_ms, 0.0, 2_500.0) {
                return Err("motion distance profile duration is invalid");
            }
        }
        if self
            .direction_duration_scales
            .values()
            .any(|value| !finite_in(*value, 0.65, 1.45))
        {
            return Err("motion direction duration scale is invalid");
        }
        self.runtime_parameters.validate()
    }

    pub fn builtin(
        fitts_a_ms: f64,
        fitts_b_ms: f64,
        side_ratio: f64,
        runtime_parameters: MotionRuntimeParameters,
    ) -> Self {
        Self {
            profile_version: 4,
            profile_id: "builtin".to_owned(),
            name: "NovaSight 内置拟人轨迹".to_owned(),
            session_id: String::new(),
            sample_count: 0,
            quality_score: 0,
            features: BTreeMap::new(),
            timing: MotionTiming {
                model: "fitts".to_owned(),
                fitts_a_ms,
                fitts_b_ms,
            },
            progress_curve: Vec::new(),
            side_offset_curve: SideCurve {
                model: "cubic_bezier_side".to_owned(),
                control_points: [0.0, side_ratio, side_ratio, 0.0],
                samples: Vec::new(),
            },
            distance_profiles: BTreeMap::new(),
            direction_duration_scales: BTreeMap::new(),
            runtime_parameters,
            profile_source: "builtin".to_owned(),
        }
    }
}

impl MotionRuntimeParameters {
    pub fn validate(&self) -> Result<(), &'static str> {
        for (value, lower, upper) in [
            (self.correction_start_ratio, 0.5, 1.0),
            (self.correction_gain, 0.0, 1.0),
            (self.side_scale, 0.0, 4.0),
            (self.max_side_ratio, 0.0, 1.0),
            (self.near_fade_start_px, 0.0, 5_000.0),
            (self.micro_bypass_px, 0.0, 1_000.0),
            (self.dynamic_rebase_ratio, 0.05, 1.0),
            (self.terminal_feedback_gain, 0.0, 2.0),
        ] {
            if !finite_in(value, lower, upper) {
                return Err("motion runtime parameter is invalid");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HumanizedMotionInput {
    pub base_x: f64,
    pub base_y: f64,
    pub full_x: f64,
    pub full_y: f64,
    pub error_x_px: f64,
    pub error_y_px: f64,
    pub target_width_px: f64,
    pub target_id: u64,
    pub trigger_active: bool,
    pub control_time_ms: f64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanizedMotionReason {
    #[default]
    DisabledOrNotTriggered,
    NonFiniteInput,
    MicroBypass,
    Active,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanizedMotionPhase {
    Startup,
    Acceleration,
    Braking,
    FineCorrection,
    ClosedLoopCorrection,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanizedSpeedCurveSource {
    MinimumJerk,
    TrainedProgress,
    Linear,
    MicroBypass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanizedSpatialCurveSource {
    CubicBezier,
    MicroBypass,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HumanizedMotionTelemetry {
    pub enabled: bool,
    pub reason: HumanizedMotionReason,
    pub phase: Option<HumanizedMotionPhase>,
    pub speed_curve_source: Option<HumanizedSpeedCurveSource>,
    pub spatial_curve_source: Option<HumanizedSpatialCurveSource>,
    pub progress: f64,
    pub side_offset: f64,
    pub planned_duration_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HumanizedMotionResult {
    pub x: f64,
    pub y: f64,
    pub telemetry: HumanizedMotionTelemetry,
}

#[derive(Clone, Debug, Default)]
pub struct HumanizedMotionGenerator {
    profile_id: Option<String>,
    target_id: Option<u64>,
    segment_start_ms: f64,
    last_progress: f64,
    initial_full_x: f64,
    initial_full_y: f64,
    initial_magnitude: f64,
    planned_duration_ms: f64,
    last_side_position: f64,
}

impl HumanizedMotionGenerator {
    pub fn reset(&mut self) {
        self.target_id = None;
        self.segment_start_ms = 0.0;
        self.last_progress = 0.0;
        self.initial_full_x = 0.0;
        self.initial_full_y = 0.0;
        self.initial_magnitude = 0.0;
        self.planned_duration_ms = 0.0;
        self.last_side_position = 0.0;
    }

    pub fn apply(
        &mut self,
        profile: Option<&MotionProfile>,
        input: HumanizedMotionInput,
    ) -> HumanizedMotionResult {
        let Some(profile) = profile else {
            self.profile_id = None;
            self.reset();
            return identity(input, HumanizedMotionReason::DisabledOrNotTriggered);
        };
        if self.profile_id.as_deref() != Some(profile.profile_id.as_str()) {
            self.profile_id = Some(profile.profile_id.clone());
            self.reset();
        }
        if !input.trigger_active {
            self.reset();
            return identity(input, HumanizedMotionReason::DisabledOrNotTriggered);
        }
        if ![
            input.base_x,
            input.base_y,
            input.full_x,
            input.full_y,
            input.error_x_px,
            input.error_y_px,
            input.control_time_ms,
        ]
        .into_iter()
        .all(f64::is_finite)
        {
            self.reset();
            return identity(input, HumanizedMotionReason::NonFiniteInput);
        }
        let params = &profile.runtime_parameters;
        let error_magnitude = input.error_x_px.hypot(input.error_y_px);
        if error_magnitude <= params.micro_bypass_px {
            self.reset();
            return HumanizedMotionResult {
                x: input.base_x,
                y: input.base_y,
                telemetry: HumanizedMotionTelemetry {
                    enabled: true,
                    reason: HumanizedMotionReason::MicroBypass,
                    phase: Some(HumanizedMotionPhase::ClosedLoopCorrection),
                    speed_curve_source: Some(HumanizedSpeedCurveSource::MicroBypass),
                    spatial_curve_source: Some(HumanizedSpatialCurveSource::MicroBypass),
                    progress: 1.0,
                    side_offset: 0.0,
                    planned_duration_ms: 0.0,
                },
            };
        }
        let current_magnitude = input.full_x.hypot(input.full_y);
        let rebase = self.target_id != Some(input.target_id)
            || self.initial_magnitude <= 1e-6
            || input.control_time_ms < self.segment_start_ms
            || direction_reversed(
                self.initial_full_x,
                self.initial_full_y,
                input.full_x,
                input.full_y,
            )
            || current_magnitude > self.initial_magnitude * (1.0 + params.dynamic_rebase_ratio)
            || endpoint_shift_ratio(
                self.initial_full_x,
                self.initial_full_y,
                input.full_x,
                input.full_y,
            ) > params.dynamic_rebase_ratio;
        if rebase {
            self.begin_segment(profile, input);
        }

        let elapsed_ms = (input.control_time_ms - self.segment_start_ms).max(0.0);
        let progress = (elapsed_ms / self.planned_duration_ms.max(1.0)).clamp(0.0, 1.0);
        let curve = progress_curve(profile, error_magnitude);
        let position = interpolate(&curve, progress);
        let previous_position = interpolate(&curve, self.last_progress);
        let delta_position = (position - previous_position).max(0.0);
        let mut side_position = side_position(profile, error_magnitude, progress);
        if !params.spatial_curve_enabled {
            side_position = 0.0;
        }
        side_position = (side_position * params.side_scale)
            .clamp(-params.max_side_ratio, params.max_side_ratio);
        if params.near_fade_start_px > 0.0 && error_magnitude < params.near_fade_start_px {
            side_position *= error_magnitude / params.near_fade_start_px;
        }
        let delta_side = side_position - self.last_side_position;
        self.last_side_position = side_position;
        self.last_progress = self.last_progress.max(progress);

        let correction_weight = if progress <= params.correction_start_ratio {
            0.0
        } else {
            ((progress - params.correction_start_ratio)
                / (1.0 - params.correction_start_ratio).max(1e-6))
            .min(1.0)
        };
        let (x, y) = if progress >= 1.0 {
            (
                input.base_x * params.terminal_feedback_gain,
                input.base_y * params.terminal_feedback_gain,
            )
        } else {
            let initial_unit_x = self.initial_full_x / self.initial_magnitude.max(1e-6);
            let initial_unit_y = self.initial_full_y / self.initial_magnitude.max(1e-6);
            let remaining_along =
                (input.full_x * initial_unit_x + input.full_y * initial_unit_y).max(0.0);
            let observed = self.initial_magnitude - remaining_along;
            let desired = self.initial_magnitude * position;
            let trajectory_error = (desired - observed).max(0.0);
            let step_cap = self.initial_magnitude * (delta_position * 1.5).max(0.02);
            let step = trajectory_error.min(step_cap);
            let direction_magnitude = current_magnitude.max(1e-6);
            let ux = input.full_x / direction_magnitude;
            let uy = input.full_y / direction_magnitude;
            let side_step = self.initial_magnitude * delta_side;
            (
                ux * step - uy * side_step
                    + input.base_x * params.correction_gain * correction_weight,
                uy * step
                    + ux * side_step
                    + input.base_y * params.correction_gain * correction_weight,
            )
        };
        HumanizedMotionResult {
            x,
            y,
            telemetry: HumanizedMotionTelemetry {
                enabled: true,
                reason: HumanizedMotionReason::Active,
                phase: Some(if progress >= 1.0 {
                    HumanizedMotionPhase::ClosedLoopCorrection
                } else {
                    motion_phase(progress)
                }),
                speed_curve_source: Some(speed_curve_source(profile, error_magnitude)),
                spatial_curve_source: Some(HumanizedSpatialCurveSource::CubicBezier),
                progress,
                side_offset: side_position,
                planned_duration_ms: self.planned_duration_ms,
            },
        }
    }

    fn begin_segment(&mut self, profile: &MotionProfile, input: HumanizedMotionInput) {
        self.target_id = Some(input.target_id);
        self.segment_start_ms = input.control_time_ms.max(0.0);
        self.last_progress = 0.0;
        self.initial_full_x = input.full_x;
        self.initial_full_y = input.full_y;
        self.initial_magnitude = input.full_x.hypot(input.full_y);
        let width = input.target_width_px.max(1.0);
        let duration = (profile.timing.fitts_a_ms
            + profile.timing.fitts_b_ms
                * (input.error_x_px.hypot(input.error_y_px) / width + 1.0).log2())
            * direction_scale(profile, input.error_x_px, input.error_y_px);
        self.planned_duration_ms = duration.clamp(35.0, 1_200.0);
        self.last_side_position = 0.0;
    }
}

fn identity(input: HumanizedMotionInput, reason: HumanizedMotionReason) -> HumanizedMotionResult {
    HumanizedMotionResult {
        x: input.base_x,
        y: input.base_y,
        telemetry: HumanizedMotionTelemetry {
            reason,
            ..HumanizedMotionTelemetry::default()
        },
    }
}

fn speed_curve_source(profile: &MotionProfile, distance: f64) -> HumanizedSpeedCurveSource {
    let raw = distance_profile(profile, distance)
        .map(|value| &value.progress_curve)
        .filter(|value| value.len() >= 2)
        .unwrap_or(&profile.progress_curve);
    if raw.len() >= 2 {
        HumanizedSpeedCurveSource::TrainedProgress
    } else if profile.runtime_parameters.minimum_jerk_fallback {
        HumanizedSpeedCurveSource::MinimumJerk
    } else {
        HumanizedSpeedCurveSource::Linear
    }
}

fn motion_phase(progress: f64) -> HumanizedMotionPhase {
    if progress < 0.2 {
        HumanizedMotionPhase::Startup
    } else if progress < 0.55 {
        HumanizedMotionPhase::Acceleration
    } else if progress < 0.78 {
        HumanizedMotionPhase::Braking
    } else {
        HumanizedMotionPhase::FineCorrection
    }
}

fn progress_curve(profile: &MotionProfile, distance: f64) -> Vec<f64> {
    let raw = distance_profile(profile, distance)
        .map(|value| &value.progress_curve)
        .filter(|value| value.len() >= 2)
        .unwrap_or(&profile.progress_curve);
    if raw.len() >= 2 {
        let mut values = raw
            .iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect::<Vec<_>>();
        values[0] = 0.0;
        for index in 1..values.len() {
            values[index] = values[index].max(values[index - 1]);
        }
        *values.last_mut().expect("curve is non-empty") = 1.0;
        return values;
    }
    if profile.runtime_parameters.minimum_jerk_fallback {
        return (0..MINIMUM_JERK_POINTS)
            .map(|index| {
                let t = index as f64 / (MINIMUM_JERK_POINTS - 1) as f64;
                10.0 * t.powi(3) - 15.0 * t.powi(4) + 6.0 * t.powi(5)
            })
            .collect();
    }
    vec![0.0, 1.0]
}

fn side_position(profile: &MotionProfile, distance: f64, progress: f64) -> f64 {
    let curve = distance_profile(profile, distance)
        .map_or(&profile.side_offset_curve, |value| &value.side_offset_curve);
    bezier(curve.control_points, progress.clamp(0.0, 1.0))
}

fn distance_profile(profile: &MotionProfile, distance: f64) -> Option<&DistanceProfile> {
    let key = if distance < 50.0 {
        "micro"
    } else if distance < 150.0 {
        "near"
    } else if distance < 350.0 {
        "mid"
    } else {
        "far"
    };
    profile.distance_profiles.get(key)
}

fn direction_scale(profile: &MotionProfile, x: f64, y: f64) -> f64 {
    let key = if x.abs() >= y.abs() * 1.5 {
        if x >= 0.0 { "right" } else { "left" }
    } else if y.abs() >= x.abs() * 1.5 {
        if y >= 0.0 { "down" } else { "up" }
    } else {
        match (y >= 0.0, x >= 0.0) {
            (true, true) => "down_right",
            (true, false) => "down_left",
            (false, true) => "up_right",
            (false, false) => "up_left",
        }
    };
    profile
        .direction_duration_scales
        .get(key)
        .copied()
        .unwrap_or(1.0)
        .clamp(0.65, 1.45)
}

fn interpolate(curve: &[f64], progress: f64) -> f64 {
    let position = progress.clamp(0.0, 1.0) * (curve.len() - 1) as f64;
    let left = (position.floor() as usize).min(curve.len() - 1);
    let right = (left + 1).min(curve.len() - 1);
    let weight = position - left as f64;
    curve[left] + (curve[right] - curve[left]) * weight
}

fn bezier(points: [f64; 4], t: f64) -> f64 {
    let u = 1.0 - t;
    u.powi(3) * points[0]
        + 3.0 * u.powi(2) * t * points[1]
        + 3.0 * u * t.powi(2) * points[2]
        + t.powi(3) * points[3]
}

fn direction_reversed(initial_x: f64, initial_y: f64, x: f64, y: f64) -> bool {
    let initial = initial_x.hypot(initial_y);
    let current = x.hypot(y);
    initial.min(current) > 1e-6 && (initial_x * x + initial_y * y) / (initial * current) < 0.25
}

fn endpoint_shift_ratio(initial_x: f64, initial_y: f64, x: f64, y: f64) -> f64 {
    let magnitude = initial_x.hypot(initial_y);
    if magnitude <= 1e-6 {
        0.0
    } else {
        (x * (-initial_y / magnitude) + y * (initial_x / magnitude)).abs() / magnitude
    }
}

fn validate_curve(curve: &[f64]) -> Result<(), &'static str> {
    if !curve.is_empty()
        && (curve.len() < 2
            || curve.len() > 128
            || curve.iter().any(|value| !finite_in(*value, 0.0, 1.0)))
    {
        return Err("motion progress curve is invalid");
    }
    Ok(())
}

fn validate_side_curve(curve: &SideCurve) -> Result<(), &'static str> {
    if curve.model != "cubic_bezier_side"
        || curve
            .control_points
            .iter()
            .any(|value| !finite_in(*value, -1.0, 1.0))
        || curve.control_points[0].abs() > 1e-9
        || curve.control_points[3].abs() > 1e-9
    {
        return Err("motion side curve is invalid");
    }
    Ok(())
}

fn finite_in(value: f64, lower: f64, upper: f64) -> bool {
    value.is_finite() && (lower..=upper).contains(&value)
}

fn default_timing_model() -> String {
    "fitts".to_owned()
}
fn default_side_model() -> String {
    "cubic_bezier_side".to_owned()
}
const fn default_fitts_a_ms() -> f64 {
    90.0
}
const fn default_fitts_b_ms() -> f64 {
    85.0
}
const fn default_correction_start_ratio() -> f64 {
    0.82
}
const fn default_correction_gain() -> f64 {
    0.35
}
const fn default_true() -> bool {
    true
}
const fn default_one() -> f64 {
    1.0
}
const fn default_max_side_ratio() -> f64 {
    0.10
}
const fn default_near_fade_start_px() -> f64 {
    24.0
}
const fn default_micro_bypass_px() -> f64 {
    3.0
}
const fn default_dynamic_rebase_ratio() -> f64 {
    0.25
}
const fn default_terminal_feedback_gain() -> f64 {
    0.35
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> MotionProfile {
        MotionProfile::builtin(35.0, 55.0, 0.012, MotionRuntimeParameters::default())
    }

    fn input(time_ms: f64, target_id: u64) -> HumanizedMotionInput {
        HumanizedMotionInput {
            base_x: 80.0,
            base_y: 0.0,
            full_x: 100.0,
            full_y: 0.0,
            error_x_px: 100.0,
            error_y_px: 0.0,
            target_width_px: 40.0,
            target_id,
            trigger_active: true,
            control_time_ms: time_ms,
        }
    }

    #[test]
    fn disabled_is_exact_identity() {
        let mut generator = HumanizedMotionGenerator::default();
        let result = generator.apply(None, input(10.0, 1));
        assert_eq!((result.x, result.y), (80.0, 0.0));
        assert!(!result.telemetry.enabled);
    }

    #[test]
    fn builtin_uses_time_curve_and_rebases_on_target_switch() {
        let profile = profile();
        assert!(profile.validate().is_ok());
        let mut generator = HumanizedMotionGenerator::default();
        let first = generator.apply(Some(&profile), input(10.0, 1));
        let second = generator.apply(Some(&profile), input(40.0, 1));
        let switched = generator.apply(Some(&profile), input(50.0, 2));
        assert_eq!(first.x, 0.0);
        assert!(second.x > 0.0);
        assert!(second.telemetry.progress > first.telemetry.progress);
        assert_eq!(switched.x, 0.0);
    }
}
