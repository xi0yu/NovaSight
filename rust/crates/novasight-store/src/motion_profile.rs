use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use novasight_core::control::humanized_motion::{
    DistanceProfile, MotionProfile, MotionRuntimeParameters, MotionTiming, SideCurve,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

const CURVE_POINTS: usize = 16;
const MAX_SAMPLE_POINTS: usize = 8_192;
const MAX_SESSION_SAMPLES: usize = 2_048;
const MAX_REPOSITORY_FILE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct MotionProfileRepository {
    root: PathBuf,
    mutation: Arc<Mutex<()>>,
}

#[derive(Debug, Error)]
pub enum MotionProfileError {
    #[error("motion training session not found")]
    SessionNotFound,
    #[error("motion profile not found")]
    ProfileNotFound,
    #[error("motion sample requires at least two points")]
    TooFewPoints,
    #[error("motion training session reached its sample limit")]
    SessionFull,
    #[error("motion repository document exceeds the 64 MiB safety limit")]
    DocumentTooLarge,
    #[error("no valid motion samples")]
    NoValidSamples,
    #[error("valid samples do not contain a usable trajectory")]
    NoUsableTrajectory,
    #[error("invalid motion identifier")]
    InvalidIdentifier,
    #[error("invalid motion profile: {0}")]
    InvalidProfile(&'static str),
    #[error("motion repository I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("motion repository JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotionSessionSummary {
    pub session_id: String,
    pub name: String,
    pub created_at: f64,
    pub sample_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MotionSession {
    session_id: String,
    name: String,
    created_at: f64,
    samples: Vec<StoredMotionSample>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct MotionPoint {
    pub t_us: i64,
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub dx: f64,
    #[serde(default)]
    pub dy: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MotionCapture {
    #[serde(default)]
    pub event_count: u64,
    #[serde(default)]
    pub coalesced_event_count: u64,
    #[serde(default)]
    pub dropped_event_count: u64,
    #[serde(default)]
    pub max_event_gap_ms: f64,
    #[serde(default)]
    pub max_dispatch_delay_ms: f64,
    #[serde(default)]
    pub boundary_hit_count: u64,
    #[serde(default)]
    pub miss_click_count: u64,
    #[serde(default)]
    pub canvas_width: u32,
    #[serde(default)]
    pub canvas_height: u32,
    #[serde(default)]
    pub device_pixel_ratio: f64,
    #[serde(default)]
    pub planned_distance_group: String,
    #[serde(default)]
    pub planned_direction: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotionSampleInput {
    #[serde(default)]
    pub sample_id: String,
    #[serde(default)]
    pub spawn_x: Option<f64>,
    #[serde(default)]
    pub spawn_y: Option<f64>,
    #[serde(default)]
    pub target_x: f64,
    #[serde(default)]
    pub target_y: f64,
    #[serde(default = "default_radius")]
    pub radius_px: f64,
    #[serde(default)]
    pub target_spawn_us: Option<i64>,
    #[serde(default)]
    pub first_motion_us: Option<i64>,
    #[serde(default)]
    pub click_us: Option<i64>,
    #[serde(default)]
    pub points: Vec<MotionPoint>,
    #[serde(default)]
    pub capture: MotionCapture,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotionSampleResult {
    pub sample_id: String,
    pub quality: String,
    pub quality_reasons: Vec<String>,
    pub metrics: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredMotionSample {
    sample_id: String,
    spawn_x: f64,
    spawn_y: f64,
    target_x: f64,
    target_y: f64,
    radius_px: f64,
    target_spawn_us: i64,
    first_motion_us: Option<i64>,
    click_us: i64,
    points: Vec<MotionPoint>,
    capture: MotionCapture,
    quality: String,
    quality_reasons: Vec<String>,
    metrics: BTreeMap<String, f64>,
}

#[derive(Clone, Debug)]
struct SampleFeatures {
    reaction_time_ms: f64,
    movement_duration_ms: f64,
    index_of_difficulty: f64,
    peak_speed_px_ms: f64,
    path_efficiency: f64,
    click_error_px: f64,
    progress_curve: Vec<f64>,
    side_offset_curve: Vec<f64>,
    correction_start_ratio: f64,
    distance_group: String,
    direction_group: String,
}

impl MotionProfileRepository {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, MotionProfileError> {
        let root = root.into();
        fs::create_dir_all(root.join("sessions"))?;
        fs::create_dir_all(root.join("profiles"))?;
        Ok(Self {
            root,
            mutation: Arc::new(Mutex::new(())),
        })
    }

    pub fn create_session(&self, name: &str) -> Result<MotionSessionSummary, MotionProfileError> {
        let _guard = self
            .mutation
            .lock()
            .unwrap_or_else(|value| value.into_inner());
        let session = MotionSession {
            session_id: new_id("session"),
            name: bounded_name(name, "未命名训练"),
            created_at: unix_seconds(),
            samples: Vec::new(),
        };
        atomic_write(&self.session_path(&session.session_id)?, &session)?;
        Ok(session.summary())
    }

    pub fn list_sessions(&self) -> Result<Vec<MotionSessionSummary>, MotionProfileError> {
        let mut sessions = read_json_dir::<MotionSession>(&self.root.join("sessions"))?
            .into_iter()
            .map(|value| value.summary())
            .collect::<Vec<_>>();
        sessions.sort_by(|left, right| right.created_at.total_cmp(&left.created_at));
        Ok(sessions)
    }

    pub fn add_sample(
        &self,
        session_id: &str,
        input: MotionSampleInput,
    ) -> Result<MotionSampleResult, MotionProfileError> {
        let _guard = self
            .mutation
            .lock()
            .unwrap_or_else(|value| value.into_inner());
        let path = self.session_path(session_id)?;
        let mut session: MotionSession = read_required(&path, MotionProfileError::SessionNotFound)?;
        if session.samples.len() >= MAX_SESSION_SAMPLES {
            return Err(MotionProfileError::SessionFull);
        }
        let sample = normalize_sample(input)?;
        let result = sample.result();
        session.samples.push(sample);
        atomic_write(&path, &session)?;
        Ok(result)
    }

    pub fn list_profiles(&self) -> Result<Vec<MotionProfile>, MotionProfileError> {
        let mut profiles = read_json_dir::<MotionProfile>(&self.root.join("profiles"))?
            .into_iter()
            .filter(|profile| profile.validate().is_ok())
            .collect::<Vec<_>>();
        profiles.sort_by(|left, right| right.profile_id.cmp(&left.profile_id));
        Ok(profiles)
    }

    pub fn profile(&self, profile_id: &str) -> Result<MotionProfile, MotionProfileError> {
        let path = self.profile_path(profile_id)?;
        let profile: MotionProfile = read_required(&path, MotionProfileError::ProfileNotFound)?;
        profile
            .validate()
            .map_err(MotionProfileError::InvalidProfile)?;
        Ok(profile)
    }

    pub fn train_profile(
        &self,
        session_id: &str,
        name: &str,
    ) -> Result<MotionProfile, MotionProfileError> {
        let _guard = self
            .mutation
            .lock()
            .unwrap_or_else(|value| value.into_inner());
        let session: MotionSession = read_required(
            &self.session_path(session_id)?,
            MotionProfileError::SessionNotFound,
        )?;
        let features = session
            .samples
            .iter()
            .filter(|sample| sample.quality == "valid")
            .filter_map(analyze_sample)
            .collect::<Vec<_>>();
        if !session
            .samples
            .iter()
            .any(|sample| sample.quality == "valid")
        {
            return Err(MotionProfileError::NoValidSamples);
        }
        if features.is_empty() {
            return Err(MotionProfileError::NoUsableTrajectory);
        }
        let (fitts_a_ms, fitts_b_ms) = fit_fitts(&features);
        let progress_curve = median_curve(
            features.iter().map(|value| value.progress_curve.as_slice()),
            true,
        );
        let side_samples = median_curve(
            features
                .iter()
                .map(|value| value.side_offset_curve.as_slice()),
            false,
        );
        let mut distance_profiles = BTreeMap::new();
        for group in ["micro", "near", "mid", "far"] {
            let items = features
                .iter()
                .filter(|value| value.distance_group == group)
                .collect::<Vec<_>>();
            if items.is_empty() {
                continue;
            }
            distance_profiles.insert(
                group.to_owned(),
                DistanceProfile {
                    sample_count: items.len(),
                    progress_curve: median_curve(
                        items.iter().map(|value| value.progress_curve.as_slice()),
                        true,
                    ),
                    side_offset_curve: fit_side_bezier(&median_curve(
                        items.iter().map(|value| value.side_offset_curve.as_slice()),
                        false,
                    )),
                    median_duration_ms: median(
                        items.iter().map(|value| value.movement_duration_ms),
                    ),
                },
            );
        }
        let directions = features
            .iter()
            .map(|value| value.direction_group.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let distances = features
            .iter()
            .map(|value| value.distance_group.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let quality_score = (25.0
            + (features.len() as f64 * 0.7).min(35.0)
            + directions as f64 * 3.0
            + distances as f64 * 4.0)
            .min(100.0) as u32;
        let terminal_gain = median(
            features
                .iter()
                .map(|value| terminal_curve_gain(&value.progress_curve)),
        );
        let mut feature_map = BTreeMap::new();
        feature_map.insert(
            "median_reaction_ms".to_owned(),
            median(features.iter().map(|value| value.reaction_time_ms)),
        );
        feature_map.insert(
            "median_duration_ms".to_owned(),
            median(features.iter().map(|value| value.movement_duration_ms)),
        );
        feature_map.insert(
            "median_peak_speed_px_ms".to_owned(),
            median(features.iter().map(|value| value.peak_speed_px_ms)),
        );
        feature_map.insert(
            "median_path_efficiency".to_owned(),
            median(features.iter().map(|value| value.path_efficiency)),
        );
        feature_map.insert("direction_coverage".to_owned(), directions as f64);
        feature_map.insert("distance_coverage".to_owned(), distances as f64);
        feature_map.insert(
            "rejected_sample_count".to_owned(),
            session.samples.len().saturating_sub(features.len()) as f64,
        );
        let profile = MotionProfile {
            profile_version: 4,
            profile_id: new_id("profile"),
            name: bounded_name(name, "真人画像"),
            session_id: session_id.to_owned(),
            sample_count: features.len(),
            quality_score,
            features: feature_map,
            timing: MotionTiming {
                model: "fitts".to_owned(),
                fitts_a_ms,
                fitts_b_ms,
            },
            progress_curve,
            side_offset_curve: fit_side_bezier(&side_samples),
            distance_profiles,
            direction_duration_scales: direction_duration_scales(&features, fitts_a_ms, fitts_b_ms),
            runtime_parameters: MotionRuntimeParameters {
                correction_start_ratio: median(
                    features.iter().map(|value| value.correction_start_ratio),
                )
                .clamp(0.55, 0.95),
                correction_gain: terminal_gain.clamp(0.15, 0.70),
                ..MotionRuntimeParameters::default()
            },
            profile_source: "trained".to_owned(),
        };
        profile
            .validate()
            .map_err(MotionProfileError::InvalidProfile)?;
        atomic_write(&self.profile_path(&profile.profile_id)?, &profile)?;
        Ok(profile)
    }

    fn session_path(&self, id: &str) -> Result<PathBuf, MotionProfileError> {
        validate_id(id, "session")?;
        Ok(self.root.join("sessions").join(format!("{id}.json")))
    }

    fn profile_path(&self, id: &str) -> Result<PathBuf, MotionProfileError> {
        validate_id(id, "profile")?;
        Ok(self.root.join("profiles").join(format!("{id}.json")))
    }
}

impl MotionSession {
    fn summary(&self) -> MotionSessionSummary {
        MotionSessionSummary {
            session_id: self.session_id.clone(),
            name: self.name.clone(),
            created_at: self.created_at,
            sample_count: self.samples.len(),
        }
    }
}

impl StoredMotionSample {
    fn result(&self) -> MotionSampleResult {
        MotionSampleResult {
            sample_id: self.sample_id.clone(),
            quality: self.quality.clone(),
            quality_reasons: self.quality_reasons.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

fn normalize_sample(input: MotionSampleInput) -> Result<StoredMotionSample, MotionProfileError> {
    let mut points = Vec::with_capacity(input.points.len().min(MAX_SAMPLE_POINTS));
    let mut last_t = -1_i64;
    let mut last_x = 0.0;
    let mut last_y = 0.0;
    for mut point in input.points.into_iter().take(MAX_SAMPLE_POINTS) {
        if !point.x.is_finite() || !point.y.is_finite() {
            continue;
        }
        point.t_us = point.t_us.max(last_t.saturating_add(1));
        if !point.dx.is_finite() {
            point.dx = if points.is_empty() {
                0.0
            } else {
                point.x - last_x
            };
        }
        if !point.dy.is_finite() {
            point.dy = if points.is_empty() {
                0.0
            } else {
                point.y - last_y
            };
        }
        last_t = point.t_us;
        last_x = point.x;
        last_y = point.y;
        points.push(point);
    }
    if points.len() < 2 {
        return Err(MotionProfileError::TooFewPoints);
    }
    let first_motion_us = input
        .first_motion_us
        .or_else(|| detect_first_motion(&points));
    let target_spawn_us = input.target_spawn_us.unwrap_or(points[0].t_us);
    let click_us = input.click_us.unwrap_or(last_t).max(last_t);
    let mut sample = StoredMotionSample {
        sample_id: if input.sample_id.is_empty() {
            new_id("sample")
        } else {
            bounded_name(&input.sample_id, "sample")
        },
        spawn_x: finite_or(input.spawn_x, points[0].x),
        spawn_y: finite_or(input.spawn_y, points[0].y),
        target_x: finite(input.target_x, 0.0),
        target_y: finite(input.target_y, 0.0),
        radius_px: finite(input.radius_px, 1.0).max(1.0),
        target_spawn_us,
        first_motion_us,
        click_us,
        points,
        capture: normalize_capture(input.capture),
        quality: String::new(),
        quality_reasons: Vec::new(),
        metrics: BTreeMap::new(),
    };
    let features = analyze_sample(&sample);
    sample.quality_reasons = quality_reasons(&sample, features.as_ref());
    sample.quality = if sample.quality_reasons.is_empty() {
        "valid"
    } else {
        "low_quality"
    }
    .to_owned();
    sample.metrics.insert(
        "max_event_gap_ms".to_owned(),
        sample.capture.max_event_gap_ms,
    );
    sample.metrics.insert(
        "max_dispatch_delay_ms".to_owned(),
        sample.capture.max_dispatch_delay_ms,
    );
    if let Some(value) = features {
        for (name, metric) in [
            ("reaction_time_ms", value.reaction_time_ms),
            ("movement_duration_ms", value.movement_duration_ms),
            ("path_efficiency", value.path_efficiency),
            ("peak_speed_px_ms", value.peak_speed_px_ms),
            ("click_error_px", value.click_error_px),
        ] {
            sample.metrics.insert(name.to_owned(), metric);
        }
    }
    Ok(sample)
}

fn analyze_sample(sample: &StoredMotionSample) -> Option<SampleFeatures> {
    if sample.points.len() < 3 {
        return None;
    }
    let first_motion = sample.first_motion_us?;
    let duration_ms = (sample.click_us - first_motion) as f64 / 1_000.0;
    if !(10.0..=2_500.0).contains(&duration_ms) {
        return None;
    }
    let reaction_time_ms = ((first_motion - sample.target_spawn_us) as f64 / 1_000.0).max(0.0);
    let start = sample.points[0];
    let click = *sample.points.last()?;
    let target_x = sample.target_x - start.x;
    let target_y = sample.target_y - start.y;
    let target_distance = target_x.hypot(target_y);
    let move_x = click.x - start.x;
    let move_y = click.y - start.y;
    let move_distance = move_x.hypot(move_y);
    if target_distance.min(move_distance) < 2.0 {
        return None;
    }
    let ux = move_x / move_distance;
    let uy = move_y / move_distance;
    let mut motion = vec![MotionPoint {
        t_us: first_motion,
        x: start.x,
        y: start.y,
        dx: 0.0,
        dy: 0.0,
    }];
    motion.extend(
        sample
            .points
            .iter()
            .copied()
            .filter(|point| point.t_us > first_motion),
    );
    if motion.len() < 3 {
        return None;
    }
    let mut times = Vec::with_capacity(motion.len());
    let mut alongs = Vec::with_capacity(motion.len());
    let mut sides = Vec::with_capacity(motion.len());
    let mut monotonic: f64 = 0.0;
    let mut peak_speed: f64 = 0.0;
    let mut path_length: f64 = 0.0;
    for (index, point) in motion.iter().enumerate() {
        let elapsed = (point.t_us - first_motion).max(0) as f64;
        monotonic = monotonic.max(
            (((point.x - start.x) * ux + (point.y - start.y) * uy) / move_distance).clamp(0.0, 1.0),
        );
        times.push((elapsed / ((sample.click_us - first_motion).max(1) as f64)).clamp(0.0, 1.0));
        alongs.push(monotonic);
        sides.push(((point.x - start.x) * -uy + (point.y - start.y) * ux) / move_distance);
        if index > 0 {
            let previous = motion[index - 1];
            path_length += (point.x - previous.x).hypot(point.y - previous.y);
            let mut window = index - 1;
            while window > 0 && point.t_us - motion[window].t_us < 2_000 {
                window -= 1;
            }
            let dt_ms = ((point.t_us - motion[window].t_us) as f64 / 1_000.0).max(0.25);
            peak_speed = peak_speed
                .max((point.x - motion[window].x).hypot(point.y - motion[window].y) / dt_ms);
        }
    }
    let mut progress_curve = sample_series_curve(&times, &alongs);
    progress_curve[0] = 0.0;
    progress_curve[CURVE_POINTS - 1] = 1.0;
    let mut side_offset_curve = sample_series_curve(&times, &sides);
    side_offset_curve[0] = 0.0;
    side_offset_curve[CURVE_POINTS - 1] = 0.0;
    let correction_start_ratio = progress_curve
        .iter()
        .position(|value| *value >= 0.82)
        .map_or(0.82, |index| index as f64 / (CURVE_POINTS - 1) as f64);
    Some(SampleFeatures {
        reaction_time_ms,
        movement_duration_ms: duration_ms,
        index_of_difficulty: (target_distance / (sample.radius_px * 2.0) + 1.0).log2(),
        peak_speed_px_ms: peak_speed,
        path_efficiency: (move_distance / move_distance.max(path_length)).min(1.0),
        click_error_px: (click.x - sample.target_x).hypot(click.y - sample.target_y),
        progress_curve,
        side_offset_curve,
        correction_start_ratio,
        distance_group: distance_group(target_distance).to_owned(),
        direction_group: direction_group(target_x, target_y).to_owned(),
    })
}

fn quality_reasons(sample: &StoredMotionSample, feature: Option<&SampleFeatures>) -> Vec<String> {
    let mut reasons = Vec::new();
    if sample.first_motion_us.is_none() {
        reasons.push("no_motion");
    }
    if sample.points.len() < 4 {
        reasons.push("too_few_points");
    }
    if let Some(value) = feature {
        if value.movement_duration_ms < 20.0 {
            reasons.push("movement_too_short");
        }
        if value.movement_duration_ms > 2_000.0 {
            reasons.push("movement_too_long");
        }
        if value.path_efficiency < 0.35 {
            reasons.push("inefficient_path");
        }
        if value.click_error_px > sample.radius_px * 1.2 {
            reasons.push("click_outside_target");
        }
        if value.reaction_time_ms > 2_000.0 {
            reasons.push("reaction_too_long");
        }
    } else if let Some(first) = sample.first_motion_us {
        let duration = (sample.click_us - first) as f64 / 1_000.0;
        if duration < 20.0 {
            reasons.push("movement_too_short");
        } else if duration > 2_000.0 {
            reasons.push("movement_too_long");
        } else {
            reasons.push("unusable_trajectory");
        }
    }
    if sample.capture.max_dispatch_delay_ms > 100.0 {
        reasons.push("dispatch_delay");
    }
    if sample.capture.boundary_hit_count > 3 {
        reasons.push("boundary_hits");
    }
    reasons.into_iter().map(str::to_owned).collect()
}

fn fit_fitts(features: &[SampleFeatures]) -> (f64, f64) {
    let mean_x = features
        .iter()
        .map(|value| value.index_of_difficulty)
        .sum::<f64>()
        / features.len() as f64;
    let mean_y = features
        .iter()
        .map(|value| value.movement_duration_ms)
        .sum::<f64>()
        / features.len() as f64;
    let variance = features
        .iter()
        .map(|value| (value.index_of_difficulty - mean_x).powi(2))
        .sum::<f64>();
    if variance <= 1e-6 {
        return ((mean_y * 0.45).max(20.0), (mean_y * 0.25).max(15.0));
    }
    let b = (features
        .iter()
        .map(|value| (value.index_of_difficulty - mean_x) * (value.movement_duration_ms - mean_y))
        .sum::<f64>()
        / variance)
        .clamp(15.0, 500.0);
    ((mean_y - b * mean_x).clamp(0.0, 1_000.0), b)
}

fn median_curve<'a>(curves: impl Iterator<Item = &'a [f64]>, monotonic: bool) -> Vec<f64> {
    let curves = curves.collect::<Vec<_>>();
    let mut result = (0..CURVE_POINTS)
        .map(|index| median(curves.iter().map(|curve| curve[index])))
        .collect::<Vec<_>>();
    result[0] = 0.0;
    if monotonic {
        for index in 1..result.len() {
            result[index] = result[index].clamp(result[index - 1], 1.0);
        }
        result[CURVE_POINTS - 1] = 1.0;
    } else {
        for value in &mut result {
            *value = value.clamp(-1.0, 1.0);
        }
        result[CURVE_POINTS - 1] = 0.0;
    }
    result
}

fn fit_side_bezier(samples: &[f64]) -> SideCurve {
    let mut aa = 0.0;
    let mut ab = 0.0;
    let mut bb = 0.0;
    let mut ay = 0.0;
    let mut by = 0.0;
    let denominator = samples.len().saturating_sub(1).max(1) as f64;
    for (index, value) in samples.iter().enumerate() {
        let t = index as f64 / denominator;
        let u = 1.0 - t;
        let a = 3.0 * u * u * t;
        let b = 3.0 * u * t * t;
        aa += a * a;
        ab += a * b;
        bb += b * b;
        ay += a * value;
        by += b * value;
    }
    let determinant: f64 = aa * bb - ab * ab;
    let (one, two) = if determinant.abs() <= 1e-9 {
        (0.0, 0.0)
    } else {
        (
            (ay * bb - by * ab) / determinant,
            (by * aa - ay * ab) / determinant,
        )
    };
    SideCurve {
        model: "cubic_bezier_side".to_owned(),
        control_points: [0.0, one.clamp(-1.0, 1.0), two.clamp(-1.0, 1.0), 0.0],
        samples: samples.to_vec(),
    }
}

fn direction_duration_scales(features: &[SampleFeatures], a: f64, b: f64) -> BTreeMap<String, f64> {
    [
        "left",
        "right",
        "up",
        "down",
        "up_left",
        "up_right",
        "down_left",
        "down_right",
    ]
    .into_iter()
    .map(|group| {
        let values = features
            .iter()
            .filter(|value| value.direction_group == group)
            .map(|value| value.movement_duration_ms / (a + b * value.index_of_difficulty).max(1.0));
        let values = values.collect::<Vec<_>>();
        (
            group.to_owned(),
            if values.is_empty() {
                1.0
            } else {
                median(values.into_iter()).clamp(0.65, 1.45)
            },
        )
    })
    .collect()
}

fn terminal_curve_gain(curve: &[f64]) -> f64 {
    let slopes = curve
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).max(0.0))
        .collect::<Vec<_>>();
    let peak = slopes.iter().copied().fold(0.0, f64::max);
    if peak <= 1e-9 {
        0.35
    } else {
        median(slopes.iter().rev().take(4).copied()) / peak
    }
}

fn sample_series_curve(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    (0..CURVE_POINTS)
        .map(|index| sample_series(xs, ys, index as f64 / (CURVE_POINTS - 1) as f64))
        .collect()
}

fn sample_series(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    for index in 1..xs.len() {
        if x <= xs[index] {
            let weight = (x - xs[index - 1]) / (xs[index] - xs[index - 1]).max(1e-9);
            return ys[index - 1] + (ys[index] - ys[index - 1]) * weight;
        }
    }
    *ys.last().unwrap_or(&0.0)
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn detect_first_motion(points: &[MotionPoint]) -> Option<i64> {
    let first = points.first()?;
    points
        .iter()
        .skip(1)
        .find(|point| (point.x - first.x).hypot(point.y - first.y) >= 1.5)
        .map(|point| point.t_us)
}

fn normalize_capture(mut value: MotionCapture) -> MotionCapture {
    value.max_event_gap_ms = finite(value.max_event_gap_ms, 0.0).max(0.0);
    value.max_dispatch_delay_ms = finite(value.max_dispatch_delay_ms, 0.0).max(0.0);
    value.device_pixel_ratio = finite(value.device_pixel_ratio, 0.0).max(0.0);
    value.planned_distance_group.truncate(32);
    value.planned_direction.truncate(32);
    value
}

fn distance_group(distance: f64) -> &'static str {
    if distance < 50.0 {
        "micro"
    } else if distance < 150.0 {
        "near"
    } else if distance < 350.0 {
        "mid"
    } else {
        "far"
    }
}
fn direction_group(x: f64, y: f64) -> &'static str {
    if x.abs() >= y.abs() * 1.5 {
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
    }
}

fn validate_id(value: &str, prefix: &str) -> Result<(), MotionProfileError> {
    if value.len() <= 128
        && value.starts_with(&format!("{prefix}_"))
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        Ok(())
    } else {
        Err(MotionProfileError::InvalidIdentifier)
    }
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}_{}", unix_nanos(), Uuid::new_v4().simple())
}
fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}
fn unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
fn default_radius() -> f64 {
    1.0
}
fn finite(value: f64, fallback: f64) -> f64 {
    if value.is_finite() { value } else { fallback }
}
fn finite_or(value: Option<f64>, fallback: f64) -> f64 {
    value
        .filter(|number| number.is_finite())
        .unwrap_or(fallback)
}
fn bounded_name(value: &str, fallback: &str) -> String {
    let mut value = if value.trim().is_empty() {
        fallback.to_owned()
    } else {
        value.trim().to_owned()
    };
    value.truncate(256);
    value
}

fn read_required<T: for<'de> Deserialize<'de>>(
    path: &Path,
    missing: MotionProfileError,
) -> Result<T, MotionProfileError> {
    if let Ok(metadata) = fs::metadata(path)
        && metadata.len() > MAX_REPOSITORY_FILE_BYTES
    {
        return Err(MotionProfileError::DocumentTooLarge);
    }
    match fs::read(path) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Err(missing),
        Err(error) => Err(error.into()),
    }
}

fn read_json_dir<T: for<'de> Deserialize<'de>>(root: &Path) -> Result<Vec<T>, MotionProfileError> {
    let mut result = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|value| value == "json")
            && let Ok(bytes) = fs::read(entry.path())
            && bytes.len() as u64 <= MAX_REPOSITORY_FILE_BYTES
            && let Ok(value) = serde_json::from_slice(&bytes)
        {
            result.push(value);
        }
    }
    Ok(result)
}

fn atomic_write(path: &Path, value: &impl Serialize) -> Result<(), MotionProfileError> {
    let temporary = path.with_extension(format!("json.{}.tmp", Uuid::new_v4().simple()));
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.len() as u64 > MAX_REPOSITORY_FILE_BYTES {
        return Err(MotionProfileError::DocumentTooLarge);
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    if let Err(error) = (|| -> io::Result<()> {
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        if let Some(parent) = path.parent() {
            OpenOptions::new().read(true).open(parent)?.sync_all()?;
        }
        Ok(())
    })() {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(target_x: f64, duration_us: i64) -> MotionSampleInput {
        MotionSampleInput {
            target_x,
            target_y: 100.0,
            radius_px: 20.0,
            target_spawn_us: Some(0),
            first_motion_us: Some(100_000),
            click_us: Some(100_000 + duration_us),
            points: vec![
                MotionPoint {
                    t_us: 0,
                    x: 20.0,
                    y: 100.0,
                    ..MotionPoint::default()
                },
                MotionPoint {
                    t_us: 100_000,
                    x: 22.0,
                    y: 100.0,
                    ..MotionPoint::default()
                },
                MotionPoint {
                    t_us: 100_000 + duration_us / 2,
                    x: 20.0 + (target_x - 20.0) * 0.55,
                    y: 100.0,
                    ..MotionPoint::default()
                },
                MotionPoint {
                    t_us: 100_000 + duration_us,
                    x: target_x,
                    y: 100.0,
                    ..MotionPoint::default()
                },
            ],
            ..serde_json::from_value(serde_json::json!({})).unwrap()
        }
    }

    #[test]
    fn repository_scores_samples_and_trains_a_valid_profile() {
        let root = std::env::temp_dir().join(format!("novasight-motion-test-{}", Uuid::new_v4()));
        let repository = MotionProfileRepository::open(&root).unwrap();
        let session = repository.create_session("test").unwrap();
        for (index, target_x) in [120.0, 240.0, 420.0, 620.0].into_iter().enumerate() {
            let result = repository
                .add_sample(
                    &session.session_id,
                    sample(target_x, 120_000 + index as i64 * 70_000),
                )
                .unwrap();
            assert_eq!(result.quality, "valid");
        }
        let profile = repository
            .train_profile(&session.session_id, "trained")
            .unwrap();
        assert_eq!(profile.profile_version, 4);
        assert_eq!(profile.progress_curve.len(), CURVE_POINTS);
        assert!(profile.timing.fitts_b_ms >= 15.0);
        assert!(profile.validate().is_ok());
        let loaded = repository.profile(&profile.profile_id).unwrap();
        assert_eq!(loaded.profile_id, profile.profile_id);
        assert_eq!(loaded.sample_count, profile.sample_count);
        assert_eq!(loaded.progress_curve, profile.progress_curve);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn path_traversal_identifier_is_rejected() {
        let root = std::env::temp_dir().join(format!("novasight-motion-test-{}", Uuid::new_v4()));
        let repository = MotionProfileRepository::open(&root).unwrap();
        assert!(matches!(
            repository.add_sample("../../escape", sample(120.0, 120_000)),
            Err(MotionProfileError::InvalidIdentifier)
        ));
        fs::remove_dir_all(root).unwrap();
    }
}
