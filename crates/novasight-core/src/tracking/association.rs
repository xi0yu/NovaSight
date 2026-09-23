use crate::error::AppError;
use crate::perception::types::Detection;

use super::classification;
use super::selection::euclidean;
use super::{Association, MAX_TRACK_CANDIDATES, TargetingConfig, Track};

#[derive(Clone, Copy, Debug)]
struct AssociationEdge {
    track_index: usize,
    detection_index: usize,
    cost: f64,
    identity_confidence: f64,
}

/// Deterministic bounded minimum-cost association, avoiding TrackId-order
/// greedy capture when multiple targets converge.
pub(super) fn associate(
    tracks: &[Track],
    detections: &[Detection],
    config: &TargetingConfig,
    captured_at_ns: u64,
) -> Result<Vec<Association>, AppError> {
    if detections.len() > MAX_TRACK_CANDIDATES {
        return Err(AppError::TooManyDetections {
            actual: detections.len(),
            maximum: MAX_TRACK_CANDIDATES,
        });
    }
    let mut sorted_detections = detections.to_vec();
    sorted_detections.sort_by_key(|det| (det.object_id(), det.class_id()));
    for window in sorted_detections.windows(2) {
        if window[0].object_id() == window[1].object_id() {
            return Err(AppError::DuplicateObjectId {
                object_id: window[0].object_id(),
            });
        }
    }
    let mut sorted_tracks = tracks.to_vec();
    sorted_tracks.sort_by_key(|track| track.id.0);

    let mut edges = Vec::with_capacity(sorted_tracks.len() * sorted_detections.len());
    for (track_index, track) in sorted_tracks.iter().enumerate() {
        for (detection_index, detection) in sorted_detections.iter().enumerate() {
            if let Some((cost, identity_confidence)) =
                association_edge(track, detection, config, captured_at_ns)
            {
                edges.push(AssociationEdge {
                    track_index,
                    detection_index,
                    cost,
                    identity_confidence,
                });
            }
        }
    }
    const FORBIDDEN_COST: f64 = 1_000_000.0;
    let mut matrix = vec![vec![FORBIDDEN_COST; sorted_detections.len()]; sorted_tracks.len()];
    for edge in &edges {
        matrix[edge.track_index][edge.detection_index] = edge.cost;
    }
    let mut out = Vec::with_capacity(sorted_tracks.len().min(sorted_detections.len()));
    for (track_index, detection_index) in linear_sum_assignment(&matrix) {
        let Some(edge) = edges.iter().find(|edge| {
            edge.track_index == track_index && edge.detection_index == detection_index
        }) else {
            continue;
        };
        let track = &sorted_tracks[track_index];
        let detection = &sorted_detections[detection_index];
        out.push(Association {
            track_id: track.id,
            object_id: detection.object_id(),
            center_x: detection.center_x(),
            center_y: detection.center_y(),
            confidence: detection.confidence(),
            identity_confidence: edge.identity_confidence,
        });
    }
    out.sort_by_key(|association| association.track_id.0);
    Ok(out)
}

/// Deterministic rectangular minimum-cost assignment in O(n^3). Normal matching
/// is capped at the active-track bound; the incumbent probe has one track and
/// at most MAX_TRACK_CANDIDATES detections.
fn linear_sum_assignment(costs: &[Vec<f64>]) -> Vec<(usize, usize)> {
    if costs.is_empty() || costs[0].is_empty() {
        return Vec::new();
    }
    let row_count = costs.len();
    let column_count = costs[0].len();
    debug_assert!(costs.iter().all(|row| row.len() == column_count));
    let transposed = row_count > column_count;
    let matrix: Vec<Vec<f64>> = if transposed {
        (0..column_count)
            .map(|column| (0..row_count).map(|row| costs[row][column]).collect())
            .collect()
    } else {
        costs.to_vec()
    };
    let rows = matrix.len();
    let columns = matrix[0].len();
    let mut u = vec![0.0; rows + 1];
    let mut v = vec![0.0; columns + 1];
    let mut p = vec![0_usize; columns + 1];
    let mut way = vec![0_usize; columns + 1];
    for row in 1..=rows {
        p[0] = row;
        let mut min_values = vec![f64::INFINITY; columns + 1];
        let mut used = vec![false; columns + 1];
        let mut column0 = 0;
        loop {
            used[column0] = true;
            let row0 = p[column0];
            let mut delta = f64::INFINITY;
            let mut column1 = 0;
            for column in 1..=columns {
                if used[column] {
                    continue;
                }
                let current = matrix[row0 - 1][column - 1] - u[row0] - v[column];
                if current < min_values[column] {
                    min_values[column] = current;
                    way[column] = column0;
                }
                if min_values[column] < delta {
                    delta = min_values[column];
                    column1 = column;
                }
            }
            for column in 0..=columns {
                if used[column] {
                    u[p[column]] += delta;
                    v[column] -= delta;
                } else {
                    min_values[column] -= delta;
                }
            }
            column0 = column1;
            if p[column0] == 0 {
                break;
            }
        }
        loop {
            let column1 = way[column0];
            p[column0] = p[column1];
            column0 = column1;
            if column0 == 0 {
                break;
            }
        }
    }
    let mut assignment = Vec::with_capacity(rows);
    for (column, &assigned_row) in p.iter().enumerate().take(columns + 1).skip(1) {
        if assigned_row == 0 {
            continue;
        }
        let row_index = assigned_row - 1;
        let column_index = column - 1;
        assignment.push(if transposed {
            (column_index, row_index)
        } else {
            (row_index, column_index)
        });
    }
    assignment.sort_unstable();
    assignment
}

fn association_edge(
    track: &Track,
    detection: &Detection,
    config: &TargetingConfig,
    captured_at_ns: u64,
) -> Option<(f64, f64)> {
    if captured_at_ns > 0 && track.last_seen_ns > 0 {
        let elapsed_ms = captured_at_ns.saturating_sub(track.last_seen_ns) as f64 / 1e6;
        if elapsed_ms > config.tracker_max_association_dt_ms {
            return None;
        }
    }
    let reference_height = track.height.max(f64::from(detection.height()));
    if !reference_height.is_finite() || reference_height <= 0.0 {
        return None;
    }
    let center = (detection.center_x(), detection.center_y());
    let nis = if track.kalman.prediction_valid() {
        track
            .kalman
            .measurement_nis(center.0, center.1, config.kalman)
    } else {
        track.kalman.measurement_nis_from_position(
            center.0,
            center.1,
            track.box_x + track.width * 0.5,
            track.box_y + track.height * 0.5,
            config.kalman,
        )
    };
    if !nis.is_finite() || nis > config.kalman.nis_hard_reject {
        return None;
    }
    let normalized_distance =
        euclidean(track.center_x, track.center_y, center.0, center.1) / reference_height;
    if !normalized_distance.is_finite() || normalized_distance > config.tracker_max_match_distance {
        return None;
    }
    let width_ratio = symmetric_ratio(track.width, f64::from(detection.width()))?;
    let height_ratio = symmetric_ratio(track.height, f64::from(detection.height()))?;
    if width_ratio > config.tracker_max_size_ratio || height_ratio > config.tracker_max_size_ratio {
        return None;
    }
    let position_weight = config.tracker_position_cost_weight.max(0.0);
    let iou_weight = config.tracker_iou_cost_weight.max(0.0);
    let scale_weight = config.tracker_scale_cost_weight.max(0.0);
    let class_weight = config.tracker_class_cost_weight.max(0.0);
    let total_weight = position_weight + iou_weight + scale_weight + class_weight;
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return None;
    }
    let overlap = track_detection_iou(track, detection);
    let scale_cost = (track.width / f64::from(detection.width())).ln().abs()
        + (track.height / f64::from(detection.height())).ln().abs();
    let cost = (position_weight * normalized_distance
        + iou_weight * (1.0 - overlap)
        + scale_weight * scale_cost
        + class_weight * classification::association_cost(track.class_id, detection.class_id()))
        / total_weight;
    Some((cost, (1.0 - cost).clamp(0.0, 1.0)))
}

fn symmetric_ratio(left: f64, right: f64) -> Option<f64> {
    if !left.is_finite() || !right.is_finite() || left <= 0.0 || right <= 0.0 {
        return None;
    }
    Some((left / right).max(right / left))
}

fn track_detection_iou(track: &Track, detection: &Detection) -> f64 {
    let track_left = track.box_x;
    let track_top = track.box_y;
    let track_right = track.box_x + track.width;
    let track_bottom = track.box_y + track.height;
    let detection_left = f64::from(detection.x());
    let detection_top = f64::from(detection.y());
    let detection_right = detection_left + f64::from(detection.width());
    let detection_bottom = detection_top + f64::from(detection.height());
    let intersection_width =
        (track_right.min(detection_right) - track_left.max(detection_left)).max(0.0);
    let intersection_height =
        (track_bottom.min(detection_bottom) - track_top.max(detection_top)).max(0.0);
    let intersection = intersection_width * intersection_height;
    let union = track.width * track.height
        + f64::from(detection.width()) * f64::from(detection.height())
        - intersection;
    if union > 0.0 {
        (intersection / union).clamp(0.0, 1.0)
    } else {
        0.0
    }
}
