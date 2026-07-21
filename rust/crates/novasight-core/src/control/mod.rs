use serde::Serialize;

use crate::{AppError, DetectionBatch, DeviceCommand, FrameStamp, MonotonicNanos, SelectedTarget};

/// Replay-only control result tied to the observation that produced it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ControlDecision {
    stamp: FrameStamp,
    decided_at: MonotonicNanos,
    target_object_id: u64,
    delta_x_counts: i32,
    delta_y_counts: i32,
}

impl ControlDecision {
    pub const fn stamp(&self) -> FrameStamp {
        self.stamp
    }

    pub const fn decided_at(&self) -> MonotonicNanos {
        self.decided_at
    }

    pub const fn target_object_id(&self) -> u64 {
        self.target_object_id
    }

    pub const fn delta_x_counts(&self) -> i32 {
        self.delta_x_counts
    }

    pub const fn delta_y_counts(&self) -> i32 {
        self.delta_y_counts
    }

    pub fn into_command(self) -> DeviceCommand {
        DeviceCommand {
            epoch: self.stamp.epoch,
            generation: self.stamp.generation,
            issued_at: self.decided_at,
            target_object_id: self.target_object_id,
            delta_x_counts: self.delta_x_counts,
            delta_y_counts: self.delta_y_counts,
        }
    }
}

/// Deterministic proportional controller for replay fixtures, not live production control.
#[derive(Clone, Copy, Debug)]
pub struct ProportionalReplayControl {
    gain: f32,
}

impl ProportionalReplayControl {
    pub const fn new(gain: f32) -> Self {
        Self { gain }
    }

    /// Validates gain, target provenance, monotonic time, and integer output range.
    pub fn decide(
        &self,
        batch: &DetectionBatch,
        target: &SelectedTarget,
        now_nanos: u64,
    ) -> Result<ControlDecision, AppError> {
        if !self.gain.is_finite() || self.gain < 0.0 {
            return Err(AppError::InvalidReplayGain);
        }
        let stamp = batch.stamp();
        if target.stamp != stamp
            || !batch.detections().iter().any(|detection| {
                detection.object_id() == target.object_id
                    && detection.class_id() == target.class_id
                    && detection.center_x() == target.center_x
                    && detection.center_y() == target.center_y
            })
        {
            return Err(AppError::TargetBatchMismatch);
        }
        if now_nanos < stamp.captured_at.0 {
            return Err(AppError::NonMonotonicControlTime {
                frame_nanos: stamp.captured_at.0,
                control_nanos: now_nanos,
            });
        }

        let (batch_center_x, batch_center_y) = batch.center();
        let delta_x_counts = scale_to_counts(target.center_x - batch_center_x, self.gain)?;
        let delta_y_counts = scale_to_counts(target.center_y - batch_center_y, self.gain)?;

        Ok(ControlDecision {
            stamp,
            decided_at: MonotonicNanos(now_nanos),
            target_object_id: target.object_id,
            delta_x_counts,
            delta_y_counts,
        })
    }
}

fn scale_to_counts(delta: f32, gain: f32) -> Result<i32, AppError> {
    let counts = f64::from(delta) * f64::from(gain);
    if !counts.is_finite() || counts < f64::from(i32::MIN) || counts > f64::from(i32::MAX) {
        return Err(AppError::DeviceCountOutOfRange);
    }
    Ok(counts.round() as i32)
}
