//! Thread-owned NVMM → CUDA → TensorRT → CUDA NMS composition.
//! Final detections are the only host-visible tensors on this path.

use crate::TensorRtError;
use novasight_core::{Detection, DetectionBatch, FrameStamp, MAX_DETECTIONS};
use std::{ffi::c_void, marker::PhantomData, path::Path, ptr::NonNull, rc::Rc};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuFrameConfig {
    pub abi_version: u32,
    pub width: u32,
    pub height: u32,
    pub candidates: u32,
    pub classes: u32,
    pub channels_first: u32,
    pub has_objectness: u32,
    pub input_dtype: u32,
    pub output_dtype: u32,
    pub bgr: u32,
    pub top_k: u32,
    pub cuda_graph: u32,
    pub scale: f32,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuModelConfig {
    pub engine: std::path::PathBuf,
    pub frame: GpuFrameConfig,
    pub input_name: String,
    pub output_name: String,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawDetection {
    left: f32,
    top: f32,
    width: f32,
    height: f32,
    confidence: f32,
    class_id: u32,
}
#[repr(C)]
struct RawResult {
    count: u32,
    truncated: u32,
    detections: [RawDetection; MAX_DETECTIONS],
}

impl RawResult {
    fn into_batch(
        self,
        stamp: FrameStamp,
        config: &GpuFrameConfig,
    ) -> Result<(DetectionBatch, u32), TensorRtError> {
        if self.count as usize > MAX_DETECTIONS {
            return Err(TensorRtError::InspectedInput(
                "GPU detection count exceeds ABI capacity".into(),
            ));
        }
        let detections = self.detections[..self.count as usize]
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if d.class_id >= config.classes {
                    return Err(TensorRtError::InspectedInput(
                        "GPU class exceeds model contract".into(),
                    ));
                }
                Detection::new(
                    i as u64,
                    d.class_id,
                    d.left,
                    d.top,
                    d.width,
                    d.height,
                    d.confidence,
                )
                .map_err(|e| TensorRtError::InspectedInput(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let batch = DetectionBatch::new(stamp, config.width, config.height, detections)
            .map_err(|e| TensorRtError::InspectedInput(e.to_string()))?;
        Ok((batch, self.truncated))
    }
}

#[derive(Debug)]
pub struct GpuFrame {
    handle: NonNull<c_void>,
    config: GpuFrameConfig,
    _thread_affine: PhantomData<Rc<()>>,
}

impl GpuFrame {
    pub fn new(path: &Path, config: &GpuModelConfig) -> Result<Self, TensorRtError> {
        #[cfg(all(feature = "gpu-frame", target_os = "linux", target_arch = "aarch64"))]
        {
            use std::ffi::CString;
            let path = CString::new(
                path.to_str()
                    .ok_or_else(|| TensorRtError::NonUtf8Path(path.into()))?,
            )
            .map_err(|_| TensorRtError::PathContainsNul)?;
            let input = CString::new(config.input_name.as_str())
                .map_err(|_| TensorRtError::InvalidNameUtf8)?;
            let output = CString::new(config.output_name.as_str())
                .map_err(|_| TensorRtError::InvalidNameUtf8)?;
            let mut handle = std::ptr::null_mut();
            let mut error = [0; crate::ERROR_CAPACITY];
            // SAFETY: buffers and config live for the call; native retains only its owned state.
            let code = unsafe {
                native::novasight_gpu_frame_create(
                    path.as_ptr(),
                    input.as_ptr(),
                    output.as_ptr(),
                    &config.frame,
                    &mut handle,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if code != 0 {
                return Err(TensorRtError::NativeCreate {
                    code,
                    detail: crate::error_text(&error),
                });
            }
            Ok(Self {
                handle: NonNull::new(handle).ok_or(TensorRtError::NullEngineHandle)?,
                config: config.frame,
                _thread_affine: PhantomData,
            })
        }
        #[cfg(not(all(feature = "gpu-frame", target_os = "linux", target_arch = "aarch64")))]
        {
            let _ = (path, config);
            Err(TensorRtError::InspectedInput(
                "NVMM GPU execution requires the Jetson build; CPU fallback forbidden".into(),
            ))
        }
    }

    /// # Safety
    /// `buffer` must point to a live GstBuffer containing an NVMM surface for the whole call.
    /// Its image pixels must not be modified concurrently. The call completes all GPU reads.
    pub unsafe fn process(
        &mut self,
        buffer: NonNull<c_void>,
        stamp: FrameStamp,
    ) -> Result<(DetectionBatch, u32), TensorRtError> {
        #[cfg(all(feature = "gpu-frame", target_os = "linux", target_arch = "aarch64"))]
        {
            let mut result = std::mem::MaybeUninit::<RawResult>::uninit();
            let mut error = [0; crate::ERROR_CAPACITY];
            // SAFETY: caller retains GstBuffer; this exclusive thread-affine owner is alive.
            let code = unsafe {
                native::novasight_gpu_frame_process(
                    self.handle.as_ptr(),
                    buffer.as_ptr(),
                    result.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if code != 0 {
                return Err(TensorRtError::NativeExecute {
                    code,
                    detail: crate::error_text(&error),
                });
            }
            // SAFETY: successful native execution initialized the complete fixed ABI result.
            unsafe { result.assume_init() }.into_batch(stamp, &self.config)
        }
        #[cfg(not(all(feature = "gpu-frame", target_os = "linux", target_arch = "aarch64")))]
        {
            let _ = (buffer, stamp);
            Err(TensorRtError::InspectedInput(
                "GPU frame execution unavailable".into(),
            ))
        }
    }
}

impl Drop for GpuFrame {
    fn drop(&mut self) {
        #[cfg(all(feature = "gpu-frame", target_os = "linux", target_arch = "aarch64"))]
        // SAFETY: unique native owner, never moved to another thread or freed before Drop.
        unsafe {
            native::novasight_gpu_frame_destroy(self.handle.as_ptr());
        }
    }
}

#[cfg(all(feature = "gpu-frame", target_os = "linux", target_arch = "aarch64"))]
mod native {
    use super::{GpuFrameConfig, RawResult};
    use std::ffi::{c_char, c_void};
    unsafe extern "C" {
        pub fn novasight_gpu_frame_create(
            path: *const c_char,
            input: *const c_char,
            output: *const c_char,
            config: *const GpuFrameConfig,
            out: *mut *mut c_void,
            error: *mut c_char,
            error_size: usize,
        ) -> i32;
        pub fn novasight_gpu_frame_process(
            context: *mut c_void,
            buffer: *mut c_void,
            out: *mut RawResult,
            error: *mut c_char,
            error_size: usize,
        ) -> i32;
        pub fn novasight_gpu_frame_destroy(context: *mut c_void);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_gpu_metadata_preserves_stamp_and_rejects_invalid_results() {
        assert_eq!(std::mem::size_of::<GpuFrameConfig>(), 60);
        assert_eq!(std::mem::size_of::<RawResult>(), 6152);
        let config = GpuFrameConfig {
            abi_version: 1,
            width: 256,
            height: 256,
            classes: 5,
            candidates: 1344,
            channels_first: 1,
            has_objectness: 0,
            input_dtype: 1,
            output_dtype: 1,
            bgr: 0,
            top_k: 256,
            cuda_graph: 1,
            scale: 1.0 / 255.0,
            confidence_threshold: 0.65,
            nms_threshold: 0.45,
        };
        let stamp = FrameStamp::new(novasight_core::RuntimeEpoch(7), 9, 123);
        let make = || RawResult {
            count: 1,
            truncated: 3,
            detections: [RawDetection {
                left: 10.0,
                top: 20.0,
                width: 30.0,
                height: 40.0,
                confidence: 0.9,
                class_id: 2,
            }; MAX_DETECTIONS],
        };
        let (batch, truncated) = make().into_batch(stamp, &config).unwrap();
        assert_eq!(batch.stamp(), stamp);
        assert_eq!(batch.detections()[0].class_id(), 2);
        assert_eq!(truncated, 3);
        let mut invalid = make();
        invalid.detections[0].confidence = f32::NAN;
        assert!(invalid.into_batch(stamp, &config).is_err());
        let mut invalid = make();
        invalid.count = 257;
        assert!(invalid.into_batch(stamp, &config).is_err());
    }
}
