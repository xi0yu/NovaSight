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
    let class_score = classification::preference_score(track.class_id, &config.class_priority);
    let distance = euclidean(
        track.observed_aim_x,
        track.observed_aim_y,
        observation_center.0,
        observation_center.1,
    );
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
    let size_score = ((track.width * track.height).max(0.0).sqrt() / radius).clamp(0.0, 1.0);
    let motion_score = if track.kalman.prediction_valid() {
        let (velocity_x, velocity_y) = track.kalman.velocity();
        let horizon_seconds = config.selection_motion_horizon_ms.max(0.0) / 1_000.0;
        let projected_distance = euclidean(
            track.observed_aim_x + velocity_x * horizon_seconds,
            track.observed_aim_y + velocity_y * horizon_seconds,
            observation_center.0,
            observation_center.1,
        );
        (0.5 + (distance - projected_distance) / (2.0 * radius)).clamp(0.0, 1.0)
    } else {
        0.5
    };
    (distance_weight * distance_score
        + class_weight * class_score
        + confidence_weight * f64::from(track.confidence).clamp(0.0, 1.0)
        + size_weight * size_score
        + continuity_weight * track.identity_confidence.clamp(0.0, 1.0)
        + motion_weight * motion_score)
        / total_weight
}

pub(super) fn detection_aspect_ratio(detection: &Detection) -> f64 {
    let width = f64::from(detection.width());
    let height = f64::from(detection.height());
    (width / height).max(height / width)
}
