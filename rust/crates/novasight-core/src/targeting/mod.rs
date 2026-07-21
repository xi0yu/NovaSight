use serde::{Deserialize, Serialize};

use crate::{DetectionBatch, FrameStamp};

/// Target identity and finite center copied from one admitted detection batch.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectedTarget {
    pub stamp: FrameStamp,
    pub object_id: u64,
    pub class_id: u32,
    pub center_x: f64,
    pub center_y: f64,
}

/// Replay selector choosing the candidate nearest the declared batch center.
#[derive(Clone, Copy, Debug, Default)]
pub struct NearestCenterTargeting;

impl NearestCenterTargeting {
    /// Breaks equal-distance candidates by ascending object ID, independent of input order.
    pub fn select(&self, batch: &DetectionBatch) -> Option<SelectedTarget> {
        let (batch_center_x, batch_center_y) = batch.center();
        batch
            .detections()
            .iter()
            .min_by(|left, right| {
                let left_distance = squared_distance(
                    left.center_x(),
                    left.center_y(),
                    batch_center_x,
                    batch_center_y,
                );
                let right_distance = squared_distance(
                    right.center_x(),
                    right.center_y(),
                    batch_center_x,
                    batch_center_y,
                );
                left_distance
                    .total_cmp(&right_distance)
                    .then_with(|| left.object_id().cmp(&right.object_id()))
            })
            .map(|detection| SelectedTarget {
                stamp: batch.stamp(),
                object_id: detection.object_id(),
                class_id: detection.class_id(),
                center_x: detection.center_x(),
                center_y: detection.center_y(),
            })
    }
}

fn squared_distance(x: f64, y: f64, center_x: f64, center_y: f64) -> f64 {
    let delta_x = x - center_x;
    let delta_y = y - center_y;
    delta_x.mul_add(delta_x, delta_y * delta_y)
}
