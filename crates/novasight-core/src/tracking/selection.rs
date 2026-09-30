use crate::perception::types::Detection;

use super::classification;
use super::{TargetingConfig, Track};

#[derive(Clone, Copy)]
pub(super) struct SelectionRank {
    pub nearby_distance: Option<f64>,
    score: f64,
}

impl SelectionRank {
    pub fn cmp(self, other: Self) -> std::cmp::Ordering {
        self.nearby_distance
            .is_some()
            .cmp(&other.nearby_distance.is_some())
            .then_with(|| match (self.nearby_distance, other.nearby_distance) {
                (Some(left), Some(right)) => right.total_cmp(&left),
                _ => std::cmp::Ordering::Equal,
            })
            .then_with(|| self.score.total_cmp(&other.score))
    }
}

fn rank(aim: (f64, f64), size: (f64, f64), center: (f64, f64), score: f64) -> SelectionRank {
    let distance = euclidean(aim.0, aim.1, center.0, center.1);
    // ponytail: a box-short-side neighborhood approximates local selection;
    // calibrate against recorded scenes before replacing it with an intent model.
    // This never enlarges the search circle or uses bbox containment as intent.
    let radius = size.0.min(size.1) * 0.5;
    SelectionRank {
        nearby_distance: (radius.is_finite() && radius > 0.0 && distance <= radius)
            .then_some(distance),
        score,
    }
}

pub(super) fn target_rank(
    track: &Track,
    observation_center: (f64, f64),
    config: &TargetingConfig,
) -> SelectionRank {
    rank(
        (track.observed_aim_x, track.observed_aim_y),
        (track.width, track.height),
        observation_center,
        target_score(track, observation_center, config),
    )
}

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
    score(
        track.class_id,
        track.confidence,
        aim,
        observation_center,
        config,
    )
}

/// Selection uses current observations only. Continuity belongs to association
/// and switch confirmation; velocity belongs to prediction, not merit scoring.
pub(super) fn candidate_score(
    detection: &Detection,
    aim: (f64, f64),
    observation_center: (f64, f64),
    config: &TargetingConfig,
) -> SelectionRank {
    rank(
        aim,
        (f64::from(detection.width()), f64::from(detection.height())),
        observation_center,
        score(
            detection.class_id(),
            detection.confidence(),
            aim,
            observation_center,
            config,
        ),
    )
}

fn score(
    class_id: u32,
    confidence: f32,
    aim: (f64, f64),
    observation_center: (f64, f64),
    config: &TargetingConfig,
) -> f64 {
    let class_score = classification::preference_score(class_id, config);
    let distance = euclidean(aim.0, aim.1, observation_center.0, observation_center.1);
    // Production supplies DetectionBatch::center() = (width / 2, height / 2).
    // Normalize by the observation half-diagonal, never the admission radius:
    // widening the search must not change merit among already eligible targets.
    let half_diagonal = observation_center.0.hypot(observation_center.1);
    let distance_score = if half_diagonal.is_finite() && half_diagonal > 0.0 {
        1.0 - (distance / half_diagonal).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let weights = config.selection_weights;
    let distance_weight = weights.distance.max(0.0);
    let class_weight = weights.class.max(0.0);
    let confidence_weight = weights.confidence.max(0.0);
    let total_weight = distance_weight + class_weight + confidence_weight;
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return 0.0;
    }
    (distance_weight * distance_score
        + class_weight * class_score
        + confidence_weight * f64::from(confidence).clamp(0.0, 1.0))
        / total_weight
}

pub(super) fn detection_aspect_ratio(detection: &Detection) -> f64 {
    let width = f64::from(detection.width());
    let height = f64::from(detection.height());
    (width / height).max(height / width)
}
