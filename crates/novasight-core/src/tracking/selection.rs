use crate::perception::types::Detection;

use super::classification;
use super::{TargetingConfig, Track};

pub(super) fn euclidean(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

pub(super) fn target_score(
    track: &Track,
    observation_center: (f64, f64),
    config: &TargetingConfig,
) -> f64 {
    let aim = (track.observed_aim_x, track.observed_aim_y);
    let distance = euclidean(aim.0, aim.1, observation_center.0, observation_center.1);
    let radius = config.target_fov_radius_px.max(1e-6);
    let motion_score = if track.kalman.prediction_valid() {
        let (velocity_x, velocity_y) = track.kalman.velocity();
        let horizon_seconds = config.selection_motion_horizon_ms.max(0.0) / 1_000.0;
        let projected_distance = euclidean(
            aim.0 + velocity_x * horizon_seconds,
            aim.1 + velocity_y * horizon_seconds,
            observation_center.0,
            observation_center.1,
        );
        (0.5 + (distance - projected_distance) / (2.0 * radius)).clamp(0.0, 1.0)
    } else {
        0.5
    };
    score(
        track.class_id,
        track.confidence,
        track.width,
        track.height,
        aim,
        track.identity_confidence,
        motion_score,
        observation_center,
        config,
    )
}

/// Ranking before association uses the same merit terms as a fresh track.
/// Temporal continuity and motion are neutral until a track exists.
pub(super) fn candidate_score(
    detection: &Detection,
    aim: (f64, f64),
    observation_center: (f64, f64),
    config: &TargetingConfig,
) -> f64 {
    score(
        detection.class_id(),
        detection.confidence(),
        f64::from(detection.width()),
        f64::from(detection.height()),
        aim,
        1.0,
        0.5,
        observation_center,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
fn score(
    class_id: u32,
    confidence: f32,
    width: f64,
    height: f64,
    aim: (f64, f64),
    continuity: f64,
    motion: f64,
    observation_center: (f64, f64),
    config: &TargetingConfig,
) -> f64 {
    let class_score = classification::preference_score(class_id, &config.class_priority);
    let distance = euclidean(aim.0, aim.1, observation_center.0, observation_center.1);
    let radius = config.target_fov_radius_px.max(1e-6);
    let distance_score = 1.0 - (distance / radius).clamp(0.0, 1.0);
    let weights = config.selection_weights;
    let distance_weight = weights.distance.max(0.0);
    let class_weight = weights.class.max(0.0);
    let confidence_weight = weights.confidence.max(0.0);
    let size_weight = weights.size.max(0.0);
    let continuity_weight = weights.continuity.max(0.0);
    let motion_weight = weights.motion.max(0.0);
    let total_weight = distance_weight
        + class_weight
        + confidence_weight
        + size_weight
        + continuity_weight
        + motion_weight;
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return 0.0;
    }
    let size_score = ((width * height).max(0.0).sqrt() / radius).clamp(0.0, 1.0);
    (distance_weight * distance_score
        + class_weight * class_score
        + confidence_weight * f64::from(confidence).clamp(0.0, 1.0)
        + size_weight * size_score
        + continuity_weight * continuity.clamp(0.0, 1.0)
        + motion_weight * motion.clamp(0.0, 1.0))
        / total_weight
}

pub(super) fn detection_aspect_ratio(detection: &Detection) -> f64 {
    let width = f64::from(detection.width());
    let height = f64::from(detection.height());
    (width / height).max(height / width)
}
