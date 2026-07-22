use std::mem::size_of;

use novasight_core::{
    AppError, Detection as CoreDetection, DetectionBatch, FrameStamp, Generation, MonotonicNanos,
    RuntimeEpoch,
};
use thiserror::Error;

use crate::{
    ABI_VERSION, FRAME_DETECTIONS_TRUNCATED, FRAME_HAS_UNTRACKED_OBJECT, FRAME_INFERENCE_DONE,
    FRAME_META_PTS_VALID, FrameSnapshot, MAX_DETECTIONS,
};

const DEEPSTREAM_UNTRACKED_OBJECT_ID: u64 = u64::MAX;
const DEEPSTREAM_CONFIDENCE_UNAVAILABLE: f32 = -0.1;

/// One synchronized reading taken at the pad probe: pipeline running time
/// (`gst_clock_get_time(clock) - base_time`) paired with process monotonic time.
/// It maps frame PTS into the clock domain used by freshness/control math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineClockSample {
    pipeline_running_now_ns: u64,
    monotonic_now: MonotonicNanos,
}

impl PipelineClockSample {
    pub const fn new(pipeline_running_now_ns: u64, monotonic_now: MonotonicNanos) -> Self {
        Self {
            pipeline_running_now_ns,
            monotonic_now,
        }
    }

    fn captured_at(self, frame_pts_ns: u64) -> Result<MonotonicNanos, AdmissionError> {
        let age_ns = self
            .pipeline_running_now_ns
            .checked_sub(frame_pts_ns)
            .ok_or(AdmissionError::FramePtsInFuture {
                frame_pts_ns,
                pipeline_running_now_ns: self.pipeline_running_now_ns,
            })?;
        self.monotonic_now
            .0
            .checked_sub(age_ns)
            .map(MonotonicNanos)
            .ok_or(AdmissionError::ClockMappingUnderflow {
                age_ns,
                monotonic_now_ns: self.monotonic_now.0,
            })
    }
}

/// Runtime-owned identity that cannot be inferred safely from vendor metadata.
/// `generation` must be allocated by the runtime ingress owner because vendor
/// frame counters can reset when a source reconnects inside the same epoch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmissionContext {
    pub epoch: RuntimeEpoch,
    pub generation: Generation,
    pub clock: PipelineClockSample,
    pub source_id: u32,
    pub inference_component_id: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedFrame {
    batch: DetectionBatch,
    filtered_without_detector_confidence: u32,
}

impl AdmittedFrame {
    pub fn batch(&self) -> &DetectionBatch {
        &self.batch
    }

    pub fn into_batch(self) -> DetectionBatch {
        self.batch
    }

    pub const fn filtered_without_detector_confidence(&self) -> u32 {
        self.filtered_without_detector_confidence
    }
}

/// Convert one caller-owned ABI snapshot into the only perception type that
/// can enter targeting/control. Incomplete or ambiguous metadata fails closed.
pub fn admit_snapshot(
    snapshot: &FrameSnapshot,
    context: AdmissionContext,
) -> Result<AdmittedFrame, AdmissionError> {
    if snapshot.abi_version != ABI_VERSION {
        return Err(AdmissionError::AbiVersion {
            expected: ABI_VERSION,
            actual: snapshot.abi_version,
        });
    }
    let expected_size = size_of::<FrameSnapshot>() as u32;
    if snapshot.struct_size != expected_size {
        return Err(AdmissionError::StructSize {
            expected: expected_size,
            actual: snapshot.struct_size,
        });
    }
    if snapshot.frame_num < 0 {
        return Err(AdmissionError::NegativeFrameNumber(snapshot.frame_num));
    }
    if snapshot.source_id != context.source_id {
        return Err(AdmissionError::SourceMismatch {
            expected: context.source_id,
            actual: snapshot.source_id,
        });
    }
    if !snapshot.has_flag(FRAME_INFERENCE_DONE) {
        return Err(AdmissionError::InferenceIncomplete);
    }
    if !snapshot.has_flag(FRAME_META_PTS_VALID) {
        return Err(AdmissionError::FramePtsMissing);
    }
    let captured_at = context.clock.captured_at(snapshot.frame_pts_ns)?;
    let detection_count = usize::try_from(snapshot.detection_count).unwrap_or(usize::MAX);
    if detection_count > MAX_DETECTIONS {
        return Err(AdmissionError::DetectionCount {
            actual: snapshot.detection_count,
            maximum: MAX_DETECTIONS,
        });
    }
    if snapshot.has_flag(FRAME_DETECTIONS_TRUNCATED) || snapshot.truncated_count != 0 {
        return Err(AdmissionError::Truncated {
            omitted: snapshot.truncated_count,
        });
    }
    if snapshot.invalid_object_count != 0 {
        return Err(AdmissionError::InvalidObjectMetadata {
            count: snapshot.invalid_object_count,
        });
    }
    if snapshot.has_flag(FRAME_HAS_UNTRACKED_OBJECT) {
        return Err(AdmissionError::UntrackedObjects);
    }

    let mut detections = Vec::with_capacity(detection_count);
    let mut filtered_without_detector_confidence = 0_u32;
    for detection in snapshot.detections() {
        if detection.object_id == DEEPSTREAM_UNTRACKED_OBJECT_ID {
            return Err(AdmissionError::UntrackedObjectPayload);
        }
        let class_id =
            u32::try_from(detection.class_id).map_err(|_| AdmissionError::NegativeClassId {
                object_id: detection.object_id,
                class_id: detection.class_id,
            })?;
        if detection.component_id != context.inference_component_id {
            return Err(AdmissionError::ComponentMismatch {
                object_id: detection.object_id,
                expected: context.inference_component_id,
                actual: detection.component_id,
            });
        }
        // DeepStream documents -0.1 when detector confidence is unavailable
        // (for example tracker-only or Group Rectangles output). Such objects
        // retain stable identity but are deliberately excluded from control;
        // assigning an invented confidence would change targeting semantics.
        if detection.confidence == DEEPSTREAM_CONFIDENCE_UNAVAILABLE {
            filtered_without_detector_confidence =
                filtered_without_detector_confidence.saturating_add(1);
            continue;
        }
        detections.push(CoreDetection::new(
            detection.object_id,
            class_id,
            detection.left,
            detection.top,
            detection.width,
            detection.height,
            detection.confidence,
        )?);
    }

    let batch = DetectionBatch::new(
        FrameStamp {
            epoch: context.epoch,
            generation: context.generation,
            captured_at,
        },
        snapshot.pipeline_width,
        snapshot.pipeline_height,
        detections,
    )
    .map_err(AdmissionError::Domain)?;
    Ok(AdmittedFrame {
        batch,
        filtered_without_detector_confidence,
    })
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum AdmissionError {
    #[error("DeepStream snapshot ABI version mismatch: expected {expected}, got {actual}")]
    AbiVersion { expected: u32, actual: u32 },
    #[error("DeepStream snapshot size mismatch: expected {expected}, got {actual}")]
    StructSize { expected: u32, actual: u32 },
    #[error("DeepStream frame number must be non-negative, got {0}")]
    NegativeFrameNumber(i64),
    #[error("DeepStream source mismatch: expected {expected}, got {actual}")]
    SourceMismatch { expected: u32, actual: u32 },
    #[error("DeepStream frame reached admission before inference completed")]
    InferenceIncomplete,
    #[error("DeepStream frame metadata has no valid PTS")]
    FramePtsMissing,
    #[error(
        "DeepStream frame PTS {frame_pts_ns} is ahead of pipeline running time {pipeline_running_now_ns}"
    )]
    FramePtsInFuture {
        frame_pts_ns: u64,
        pipeline_running_now_ns: u64,
    },
    #[error(
        "DeepStream frame age {age_ns} cannot be represented at process monotonic time {monotonic_now_ns}"
    )]
    ClockMappingUnderflow { age_ns: u64, monotonic_now_ns: u64 },
    #[error("DeepStream snapshot declares {actual} detections; maximum is {maximum}")]
    DetectionCount { actual: u32, maximum: usize },
    #[error("DeepStream omitted {omitted} detections because the snapshot was truncated")]
    Truncated { omitted: u32 },
    #[error("DeepStream frame contained {count} invalid object metadata entries")]
    InvalidObjectMetadata { count: u32 },
    #[error("DeepStream frame contains untracked objects; stable object IDs are required")]
    UntrackedObjects,
    #[error("DeepStream object payload uses UNTRACKED_OBJECT_ID without a consistent frame flag")]
    UntrackedObjectPayload,
    #[error("DeepStream object {object_id} has negative class ID {class_id}")]
    NegativeClassId { object_id: u64, class_id: i32 },
    #[error(
        "DeepStream object {object_id} came from component {actual}; expected component {expected}"
    )]
    ComponentMismatch {
        object_id: u64,
        expected: i32,
        actual: i32,
    },
    #[error(transparent)]
    Domain(#[from] AppError),
}

impl AdmissionError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::AbiVersion { .. } => "deepstream_abi_version_mismatch",
            Self::StructSize { .. } => "deepstream_snapshot_size_mismatch",
            Self::NegativeFrameNumber(_) => "deepstream_frame_number_invalid",
            Self::SourceMismatch { .. } => "deepstream_source_mismatch",
            Self::InferenceIncomplete => "deepstream_inference_incomplete",
            Self::FramePtsMissing => "deepstream_frame_pts_missing",
            Self::FramePtsInFuture { .. } => "deepstream_frame_pts_future",
            Self::ClockMappingUnderflow { .. } => "deepstream_clock_mapping_invalid",
            Self::DetectionCount { .. } => "deepstream_detection_count_invalid",
            Self::Truncated { .. } => "deepstream_detections_truncated",
            Self::InvalidObjectMetadata { .. } => "deepstream_object_metadata_invalid",
            Self::UntrackedObjects => "deepstream_objects_untracked",
            Self::UntrackedObjectPayload => "deepstream_object_id_untracked",
            Self::NegativeClassId { .. } => "deepstream_class_id_invalid",
            Self::ComponentMismatch { .. } => "deepstream_component_mismatch",
            Self::Domain(_) => "deepstream_detection_invalid",
        }
    }
}
