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
            return Err(TensorRtError::InspectedInput(format!(
                "GPU result: count={} exceeds ABI capacity={MAX_DETECTIONS}",
                self.count
            )));
        }
        // Validate only the bounded delivery envelope; never decode raw tensors
        // or repeat sorting/NMS on the host.
        let total = u64::from(self.count) + u64::from(self.truncated);
        let maximum =
            u64::from(config.candidates).min(u64::from(config.classes) * u64::from(config.top_k));
        if total > maximum || (self.truncated != 0 && self.count as usize != MAX_DETECTIONS) {
            return Err(TensorRtError::InspectedInput(format!(
                "GPU result: count={}, truncated={}, maximum={maximum}; inconsistent result envelope",
                self.count, self.truncated,
            )));
        }
        let detections = self.detections[..self.count as usize]
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if d.class_id >= config.classes {
                    return Err(TensorRtError::InspectedInput(format!(
                        "GPU result[{i}]: class_id={} exceeds class_count={}",
                        d.class_id, config.classes,
                    )));
                }
                if !d.confidence.is_finite()
                    || d.confidence <= 0.0
                    || d.confidence < config.confidence_threshold
                    || d.confidence > 1.0
                {
                    return Err(TensorRtError::InspectedInput(format!(
                        "GPU result[{i}]: confidence={} must be finite, positive, >= {} and <= 1",
                        d.confidence, config.confidence_threshold,
                    )));
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
                .map_err(|e| TensorRtError::InspectedInput(format!("GPU result[{i}]: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let batch = DetectionBatch::new(stamp, config.width, config.height, detections)
            .map_err(|e| TensorRtError::InspectedInput(format!("GPU result batch: {e}")))?;
        Ok((batch, self.truncated))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuTimings {
    pub abi_version: u32,
    pub warmed: u32,
    /// Whole `novasight_gpu_frame_process` host wall time (ns).
    pub total_ns: u64,
    /// gst_buffer_map + NvBufSurface checks + EGLImage register/map (ns).
    pub import_ns: u64,
    /// Host time to enqueue preprocess/TensorRT/decode-NMS work (ns).
    pub launch_ns: u64,
    /// Blocking wait until preprocess + TensorRT + GPU decode/NMS complete (ns).
    pub gpu_wait_ns: u64,
    /// Stream sync + EGL unregister/unmap + gst_buffer_unmap (ns).
    pub release_ns: u64,
    /// CUDA graph capture on the first warmed frame (ns, zero afterwards).
    pub graph_capture_ns: u64,
}

/// Successful output of one [`GpuFrame::process`] call.
pub struct GpuProcessOutput {
    pub batch: DetectionBatch,
    pub truncated: u32,
    pub timings: GpuTimings,
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
    ) -> Result<GpuProcessOutput, TensorRtError> {
        #[cfg(all(feature = "gpu-frame", target_os = "linux", target_arch = "aarch64"))]
        {
            let mut result = std::mem::MaybeUninit::<RawResult>::uninit();
            let mut timings = std::mem::MaybeUninit::<GpuTimings>::uninit();
            let mut error = [0; crate::ERROR_CAPACITY];
            // SAFETY: caller retains GstBuffer; this exclusive thread-affine owner is alive.
            let code = unsafe {
                native::novasight_gpu_frame_process(
                    self.handle.as_ptr(),
                    buffer.as_ptr(),
                    result.as_mut_ptr(),
                    timings.as_mut_ptr(),
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
            // SAFETY: successful native execution initialized the complete fixed ABI structs.
            let (batch, truncated) =
                unsafe { result.assume_init() }.into_batch(stamp, &self.config)?;
            let timings = unsafe { timings.assume_init() };
            Ok(GpuProcessOutput {
                batch,
                truncated,
                timings,
            })
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
    use super::{GpuFrameConfig, GpuTimings, RawResult};
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
            timings: *mut GpuTimings,
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
        assert_eq!(
            std::mem::size_of::<GpuTimings>(),
            2 * std::mem::size_of::<u32>() + 6 * std::mem::size_of::<u64>()
        );
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
            truncated: 0,
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
        assert_eq!(truncated, 0);
        let mut full = make();
        full.count = MAX_DETECTIONS as u32;
        full.truncated = 3;
        assert_eq!(full.into_batch(stamp, &config).unwrap().1, 3);
        let mut empty = make();
        empty.count = 0;
        let (empty, _) = empty.into_batch(stamp, &config).unwrap();
        assert!(empty.detections().is_empty());
        assert_eq!(empty.stamp(), stamp);
        let mut invalid = make();
        invalid.detections[0].confidence = f32::NAN;
        assert!(invalid.into_batch(stamp, &config).is_err());
        let mut invalid = make();
        invalid.count = 257;
        assert!(invalid.into_batch(stamp, &config).is_err());
        for (count, truncated) in [(1, 1), (0, 1), (256, u32::MAX)] {
            let mut invalid = make();
            invalid.count = count;
            invalid.truncated = truncated;
            assert!(
                invalid
                    .into_batch(stamp, &config)
                    .unwrap_err()
                    .to_string()
                    .contains("result envelope")
            );
        }
        for confidence in [0.0, -0.1, 0.64, 1.01, f32::INFINITY] {
            let mut invalid = make();
            invalid.detections[0].confidence = confidence;
            assert!(
                invalid
                    .into_batch(stamp, &config)
                    .unwrap_err()
                    .to_string()
                    .contains("GPU result[0]: confidence")
            );
        }
        let mut invalid = make();
        invalid.detections[0].class_id = config.classes;
        assert!(
            invalid
                .into_batch(stamp, &config)
                .unwrap_err()
                .to_string()
                .contains("class_id=5")
        );
        for (left, top, width, height) in [
            (-1.0, 0.0, 2.0, 2.0),
            (0.0, -1.0, 2.0, 2.0),
            (255.0, 0.0, 2.0, 2.0),
            (0.0, 255.0, 2.0, 2.0),
            (0.0, 0.0, 0.0, 2.0),
            (f32::NAN, 0.0, 2.0, 2.0),
        ] {
            let mut invalid = make();
            invalid.detections[0].left = left;
            invalid.detections[0].top = top;
            invalid.detections[0].width = width;
            invalid.detections[0].height = height;
            assert!(invalid.into_batch(stamp, &config).is_err());
        }
        let mut boundary = make();
        boundary.detections[0].confidence = config.confidence_threshold;
        boundary.detections[0].left = 226.0;
        boundary.detections[0].top = 216.0;
        assert!(boundary.into_batch(stamp, &config).is_ok());
    }
}
