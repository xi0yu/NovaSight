use std::sync::Arc;
use std::sync::mpsc::SyncSender;

use novasight_core::{Clock, RuntimeEpoch};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::PipelineIngress;

/// Stable identity passed to the perception adapter before a catalog
/// deployment is changed. This lets the adapter validate the candidate that
/// was requested instead of accidentally re-validating the currently active
/// model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelCandidate {
    pub project_id: i64,
    pub artifact_id: i64,
    pub parser_preset: String,
}

/// Runtime facts obtained from the validated engine manifest. Registry labels
/// are deliberately not used as the source of truth for these fields.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PerceptionModelContract {
    pub input_shape: String,
    pub input_width: u32,
    pub input_height: u32,
    /// DeepStream maps letterboxed detections back into its ROI-sized input;
    /// direct resize adapters report model-tensor coordinates instead.
    pub preserves_roi_coordinates: bool,
    pub classes: Vec<String>,
    pub parser: ParserContract,
}

/// Geometry resolved for one concrete capture + model runtime epoch. Candidate
/// publication deliberately returns only [`PerceptionModelContract`]; the
/// camera-dependent values are resolved immediately before startup.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PerceptionRuntimeContract {
    pub model: PerceptionModelContract,
    pub source_width: u32,
    pub roi_width: u32,
    pub roi_height: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParserContract {
    pub requested_preset: String,
    pub compatibility: String,
    pub has_objectness: bool,
    pub parser_library: String,
    pub parser_function: String,
    pub nms_owner: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PerceptionEvent {
    Faulted { message: String },
    Stopped,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PerceptionMetrics {
    pub input_buffers: u64,
    pub probed_buffers: u64,
    pub published_batches: u64,
    /// Capture timestamp of the newest DetectionBatch accepted by ingress.
    /// It shares the runtime monotonic clock domain used for freshness gates.
    #[serde(default)]
    pub latest_published_capture_at_ns: Option<u64>,
    /// Most recent correlated nvinfer sink-to-src duration. Collection is a
    /// single clock read and subtraction in the existing output pad probe;
    /// zero means no correlated sample has been observed yet.
    #[serde(default)]
    pub latest_inference_duration_ns: Option<u64>,
    #[serde(default)]
    pub inference_duration_samples: u64,
    pub busy_dropped_batches: u64,
    pub overwritten_snapshots: u64,
    pub unavailable_snapshot_slots: u64,
    pub extraction_rejections: u64,
    pub admission_rejections: u64,
    pub truncated_detections: u64,
    pub ingress_rejections: u64,
    pub timestamp_buffer_pts_matches: u64,
    pub timestamp_frame_meta_pts_matches: u64,
    pub timestamp_correlation_misses: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerceptionErrorKind {
    Other,
    ActiveModelMissing,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("perception adapter failed: {message}")]
pub struct PerceptionError {
    kind: PerceptionErrorKind,
    message: String,
}

impl PerceptionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            kind: PerceptionErrorKind::Other,
            message: message.into(),
        }
    }

    pub fn active_model_missing(message: impl Into<String>) -> Self {
        Self {
            kind: PerceptionErrorKind::ActiveModelMissing,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> PerceptionErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Normalize the Studio parser-preset vocabulary and verify its objectness
/// semantics against the immutable engine manifest.
pub fn validate_parser_preset(
    value: &str,
    has_objectness: bool,
) -> Result<String, PerceptionError> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    let preset = match normalized.as_str() {
        "" | "automatic" | "auto" => "auto",
        "yolo_v5" | "yolov5_raw" | "yolov5" => "yolov5",
        "yolo_v8" | "yolov8_raw" | "yolov8" => "yolov8",
        "yolo_11" | "yolo11_raw" | "yolo11" => "yolo11",
        "generic" | "custom" | "novasight" | "novasight_generic" => "novasight_generic",
        _ => {
            return Err(PerceptionError::new(format!(
                "unsupported parser preset {value}"
            )));
        }
    };
    let compatible = match preset {
        "auto" | "novasight_generic" => true,
        "yolov5" => has_objectness,
        "yolov8" | "yolo11" => !has_objectness,
        _ => false,
    };
    if !compatible {
        return Err(PerceptionError::new(format!(
            "parser preset {preset} conflicts with manifest objectness={has_objectness}"
        )));
    }
    Ok(preset.to_owned())
}

/// Factory for one epoch-scoped perception producer.
pub trait PerceptionAdapter: Send + Sync + 'static {
    /// Validate the next epoch's external resources without starting capture,
    /// inference, or output. Adapters with dynamic configuration should do the
    /// same resolution work here that [`Self::start`] performs.
    fn preflight(&self) -> Result<(), PerceptionError> {
        Ok(())
    }

    /// Validate a specific not-yet-active model. Production adapters must
    /// resolve `candidate.artifact_id` directly; reading the active deployment
    /// here would validate the wrong engine during a switch.
    fn preflight_model(
        &self,
        _candidate: &ModelCandidate,
    ) -> Result<Option<PerceptionModelContract>, PerceptionError> {
        self.preflight()?;
        Ok(None)
    }

    /// Resolve the active model together with the current concrete capture
    /// plan. The supervisor installs this geometry before constructing the
    /// control pipeline for a new epoch.
    fn runtime_contract(&self) -> Result<Option<PerceptionRuntimeContract>, PerceptionError> {
        Ok(None)
    }

    fn start(
        &self,
        epoch: RuntimeEpoch,
        ingress: PipelineIngress,
        clock: Arc<dyn Clock>,
        events: SyncSender<PerceptionEvent>,
    ) -> Result<Box<dyn PerceptionSession>, PerceptionError>;
}

/// Running perception resources owned by the RuntimeSupervisor.
pub trait PerceptionSession: Send + 'static {
    fn metrics(&self) -> PerceptionMetrics {
        PerceptionMetrics::default()
    }

    /// Non-blocking liveness check for the epoch-scoped producer.
    ///
    /// Event delivery remains the fast failure path, but the supervisor polls
    /// this method as an independent backstop for owner-thread termination that
    /// could not publish an event (for example, a panic during vendor code).
    fn poll_health(&mut self) -> Result<(), PerceptionError> {
        Ok(())
    }

    /// Stop producing before the post-inference pipeline is closed.
    fn shutdown(&mut self) -> Result<(), PerceptionError>;
}

#[cfg(test)]
mod tests {
    use super::validate_parser_preset;

    #[test]
    fn parser_presets_preserve_studio_vocabulary_and_objectness_semantics() {
        for preset in ["auto", "yolov8", "yolo11", "novasight_generic"] {
            assert_eq!(validate_parser_preset(preset, false).unwrap(), preset);
        }
        assert!(validate_parser_preset("yolov5", false).is_err());
        assert_eq!(validate_parser_preset("yolo_v5", true).unwrap(), "yolov5");
        assert!(validate_parser_preset("yolov8", true).is_err());
        assert!(validate_parser_preset("unknown", false).is_err());
    }
}
