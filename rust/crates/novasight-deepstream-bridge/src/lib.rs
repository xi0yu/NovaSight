#[cfg(feature = "ffi")]
use std::ffi::c_void;
#[cfg(feature = "ffi")]
use std::mem::MaybeUninit;
#[cfg(feature = "ffi")]
use std::mem::size_of;
#[cfg(feature = "ffi")]
use std::ptr::NonNull;

mod admission;

pub use admission::{
    AdmissionContext, AdmissionError, AdmittedCapture, AdmittedFrame, PipelineClockSample,
    admit_capture_snapshot, admit_snapshot,
};

pub const ABI_VERSION: u32 = 1;
pub const MAX_DETECTIONS: usize = 256;
pub const ANY_SOURCE: u32 = u32::MAX;

pub const FRAME_INFERENCE_DONE: u32 = 1 << 0;
pub const FRAME_DETECTIONS_TRUNCATED: u32 = 1 << 1;
pub const FRAME_BUFFER_PTS_VALID: u32 = 1 << 2;
pub const FRAME_META_PTS_VALID: u32 = 1 << 3;
pub const FRAME_HAS_UNTRACKED_OBJECT: u32 = 1 << 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Detection {
    pub object_id: u64,
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
    pub class_id: i32,
    pub component_id: i32,
    pub flags: u32,
}

#[repr(C)]
pub struct FrameSnapshot {
    pub abi_version: u32,
    pub struct_size: u32,
    pub frame_num: i64,
    pub buffer_pts_ns: u64,
    pub frame_pts_ns: u64,
    pub ntp_timestamp_ns: u64,
    pub source_id: u32,
    pub batch_id: u32,
    pub source_width: u32,
    pub source_height: u32,
    pub pipeline_width: u32,
    pub pipeline_height: u32,
    pub detection_count: u32,
    pub truncated_count: u32,
    pub invalid_object_count: u32,
    pub flags: u32,
    pub detections: [Detection; MAX_DETECTIONS],
}

impl Default for FrameSnapshot {
    fn default() -> Self {
        // Every field is an integer, float, or array of those; all-zero is valid.
        unsafe { std::mem::zeroed() }
    }
}

impl FrameSnapshot {
    pub fn detections(&self) -> &[Detection] {
        let count = usize::try_from(self.detection_count)
            .unwrap_or(MAX_DETECTIONS)
            .min(MAX_DETECTIONS);
        &self.detections[..count]
    }

    pub fn has_flag(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum Status {
    Ok = 0,
    InvalidArgument = 1,
    OutputTooSmall = 2,
    NoBatchMeta = 3,
    NoFrameMeta = 4,
    SourceNotFound = 5,
    InternalError = 6,
}

impl Status {
    #[cfg(feature = "ffi")]
    fn from_raw(value: i32) -> Result<Self, i32> {
        match value {
            0 => Ok(Self::Ok),
            1 => Ok(Self::InvalidArgument),
            2 => Ok(Self::OutputTooSmall),
            3 => Ok(Self::NoBatchMeta),
            4 => Ok(Self::NoFrameMeta),
            5 => Ok(Self::SourceNotFound),
            6 => Ok(Self::InternalError),
            other => Err(other),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtractError {
    pub status: Option<Status>,
    pub raw_status: i32,
}

#[cfg(feature = "ffi")]
unsafe extern "C" {
    fn ns_ds_bridge_abi_version() -> u32;
    fn ns_ds_bridge_frame_snapshot_size() -> u32;
    fn ns_ds_extract_frame(
        buffer: *mut c_void,
        source_id: u32,
        output: *mut FrameSnapshot,
        output_size: u32,
    ) -> i32;
}

#[cfg(feature = "ffi")]
pub fn validate_loaded_abi() -> Result<(), &'static str> {
    let version = unsafe { ns_ds_bridge_abi_version() };
    let frame_size = unsafe { ns_ds_bridge_frame_snapshot_size() };
    if version != ABI_VERSION {
        return Err("DeepStream bridge ABI version mismatch");
    }
    if frame_size as usize != size_of::<FrameSnapshot>() {
        return Err("DeepStream bridge FrameSnapshot size mismatch");
    }
    Ok(())
}

/// Copy DeepStream metadata into caller-owned storage during a live pad probe.
///
/// # Safety
///
/// `buffer` must point to a live `GstBuffer` for the full duration of this call.
/// The caller must invoke this synchronously inside the pad probe that borrowed
/// the buffer. The returned snapshot owns no pointer into GStreamer or
/// DeepStream and may be published to another thread after this call returns.
#[cfg(feature = "ffi")]
pub unsafe fn extract_frame_into(
    buffer: NonNull<c_void>,
    source_id: u32,
    output: &mut MaybeUninit<FrameSnapshot>,
) -> Result<&FrameSnapshot, ExtractError> {
    let raw_status = unsafe {
        ns_ds_extract_frame(
            buffer.as_ptr(),
            source_id,
            output.as_mut_ptr(),
            size_of::<FrameSnapshot>() as u32,
        )
    };
    match Status::from_raw(raw_status) {
        Ok(Status::Ok) => Ok(unsafe { output.assume_init_ref() }),
        Ok(status) => Err(ExtractError {
            status: Some(status),
            raw_status,
        }),
        Err(other) => Err(ExtractError {
            status: None,
            raw_status: other,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn rust_layout_matches_the_c_contract() {
        assert_eq!(size_of::<Detection>(), 40);
        assert_eq!(align_of::<Detection>(), 8);
        assert_eq!(offset_of!(Detection, left), 8);
        assert_eq!(offset_of!(Detection, confidence), 24);
        assert_eq!(offset_of!(Detection, class_id), 28);
        assert_eq!(offset_of!(Detection, flags), 36);
        assert_eq!(size_of::<FrameSnapshot>(), 10_320);
        assert_eq!(align_of::<FrameSnapshot>(), 8);
        assert_eq!(offset_of!(FrameSnapshot, frame_num), 8);
        assert_eq!(offset_of!(FrameSnapshot, buffer_pts_ns), 16);
        assert_eq!(offset_of!(FrameSnapshot, source_id), 40);
        assert_eq!(offset_of!(FrameSnapshot, detection_count), 64);
        assert_eq!(offset_of!(FrameSnapshot, detections), 80);
    }

    #[test]
    fn owned_snapshot_can_cross_threads_without_vendor_pointers() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FrameSnapshot>();
    }

    #[test]
    fn detection_slice_is_bounded_by_capacity() {
        let mut frame = FrameSnapshot {
            detection_count: u32::MAX,
            ..FrameSnapshot::default()
        };
        assert_eq!(frame.detections().len(), MAX_DETECTIONS);
        frame.detection_count = 3;
        assert_eq!(frame.detections().len(), 3);
    }
}
