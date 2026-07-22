#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(any(feature = "ffi", test))]
use std::ffi::{CStr, CString, c_char};
use std::marker::PhantomData;
use std::num::{NonZeroU32, NonZeroU64};
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

use novasight_core::{Generation, MonotonicNanos, RuntimeEpoch};
#[cfg(any(feature = "ffi", test))]
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(any(feature = "ffi", test))]
const ABI_VERSION: u32 = 1;
const OUTPUT_CAPACITY: usize = 16 * 1024;
const MAX_NATIVE_DIMENSION: u32 = i32::MAX as u32;
const MAX_RGBA_WIDTH: u32 = (i32::MAX / 4) as u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorDtype {
    Float16,
    Float32,
}

impl TensorDtype {
    #[cfg(any(feature = "ffi", test))]
    const fn as_abi_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::Float32 => "float32",
        }
    }

    const fn byte_width(self) -> u64 {
        match self {
            Self::Float16 => 2,
            Self::Float32 => 4,
        }
    }
}

/// Exact model input contract supported by the existing Jetson CUDA kernel:
/// batch-one RGB NCHW, direct resize, and `1/255` normalization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorContract {
    shape: [u32; 4],
    dtype: TensorDtype,
    nbytes: u64,
}

impl TensorContract {
    pub fn rgb_nchw(height: u32, width: u32, dtype: TensorDtype) -> Result<Self, PreprocessError> {
        let height = NonZeroU32::new(height).ok_or(PreprocessError::ZeroTensorDimension)?;
        let width = NonZeroU32::new(width).ok_or(PreprocessError::ZeroTensorDimension)?;
        if height.get() > MAX_NATIVE_DIMENSION || width.get() > MAX_RGBA_WIDTH {
            return Err(PreprocessError::GeometryExceedsNativeLimits {
                width: width.get(),
                height: height.get(),
            });
        }
        let nbytes = 3_u64
            .checked_mul(u64::from(height.get()))
            .and_then(|value| value.checked_mul(u64::from(width.get())))
            .and_then(|value| value.checked_mul(dtype.byte_width()))
            .ok_or(PreprocessError::TensorSizeOverflow)?;
        Ok(Self {
            shape: [1, 3, height.get(), width.get()],
            dtype,
            nbytes,
        })
    }

    pub const fn shape(self) -> [u32; 4] {
        self.shape
    }

    pub const fn dtype(self) -> TensorDtype {
        self.dtype
    }

    pub const fn nbytes(self) -> u64 {
        self.nbytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NvmmFrameInput {
    epoch: RuntimeEpoch,
    generation: Generation,
    captured_at: MonotonicNanos,
    gst_buffer_ptr: NonZeroU64,
    width: NonZeroU32,
    height: NonZeroU32,
}

impl NvmmFrameInput {
    pub fn from_deepstream_pad(
        epoch: RuntimeEpoch,
        generation: Generation,
        captured_at: MonotonicNanos,
        gst_buffer_ptr: u64,
        width: u32,
        height: u32,
    ) -> Result<Self, PreprocessError> {
        if epoch.0 == 0 {
            return Err(PreprocessError::ZeroEpoch);
        }
        if generation.0 == 0 {
            return Err(PreprocessError::ZeroGeneration);
        }
        if captured_at.0 == 0 {
            return Err(PreprocessError::ZeroCaptureTimestamp);
        }
        let width = NonZeroU32::new(width).ok_or(PreprocessError::ZeroFrameDimension)?;
        let height = NonZeroU32::new(height).ok_or(PreprocessError::ZeroFrameDimension)?;
        if width.get() > MAX_NATIVE_DIMENSION || height.get() > MAX_NATIVE_DIMENSION {
            return Err(PreprocessError::GeometryExceedsNativeLimits {
                width: width.get(),
                height: height.get(),
            });
        }
        if !width.get().is_multiple_of(2) || !height.get().is_multiple_of(2) {
            return Err(PreprocessError::OddNv12Geometry {
                width: width.get(),
                height: height.get(),
            });
        }
        Ok(Self {
            epoch,
            generation,
            captured_at,
            gst_buffer_ptr: NonZeroU64::new(gst_buffer_ptr)
                .ok_or(PreprocessError::NullGstBuffer)?,
            width,
            height,
        })
    }

    pub const fn epoch(self) -> RuntimeEpoch {
        self.epoch
    }

    pub const fn generation(self) -> Generation {
        self.generation
    }

    pub const fn captured_at(self) -> MonotonicNanos {
        self.captured_at
    }
}

#[cfg(any(feature = "ffi", test))]
#[derive(Clone)]
pub struct CudaPreprocessor {
    releases: Arc<ReleaseRegistry>,
    backend: Arc<str>,
}

#[cfg(any(feature = "ffi", test))]
impl std::fmt::Debug for CudaPreprocessor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaPreprocessor")
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

#[cfg(any(feature = "ffi", test))]
impl CudaPreprocessor {
    #[cfg(feature = "ffi")]
    pub fn linked() -> Result<Self, PreprocessError> {
        Self::from_abi(Arc::new(LinkedNativeAbi))
    }

    fn from_abi(abi: Arc<dyn NativePreprocessAbi>) -> Result<Self, PreprocessError> {
        let actual = abi.abi_version();
        if actual != ABI_VERSION {
            return Err(PreprocessError::AbiVersion {
                expected: ABI_VERSION,
                actual,
            });
        }
        let (status_code, status_json) = call_output(|output| abi.status(output))?;
        if status_code != 0 {
            return Err(PreprocessError::StatusCall(status_code));
        }
        let status: NativeStatus = serde_json::from_str(&status_json)
            .map_err(|source| PreprocessError::InvalidStatusJson { source })?;
        if status.abi_version != ABI_VERSION {
            return Err(PreprocessError::AbiVersion {
                expected: ABI_VERSION,
                actual: status.abi_version,
            });
        }
        if !status.available
            || !status.ready
            || !status.zero_copy
            || status.memory_space != "cuda_device"
            || status.backend.trim().is_empty()
        {
            return Err(PreprocessError::NotReady {
                backend: status.backend,
                reason: status
                    .reason
                    .unwrap_or_else(|| "native_not_ready".to_owned()),
                detail: status.detail.unwrap_or_default(),
            });
        }
        Ok(Self {
            releases: Arc::new(ReleaseRegistry::new(abi)),
            backend: Arc::from(status.backend),
        })
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn prepare(
        &self,
        frame: NvmmFrameInput,
        contract: TensorContract,
    ) -> Result<DeviceTensor, PreprocessError> {
        self.releases.retry_pending();
        let payload = PreparePayload::new(frame, contract);
        let payload = serde_json::to_vec(&payload)
            .map_err(|source| PreprocessError::SerializeRequest { source })?;
        let payload = CString::new(payload).map_err(|_| PreprocessError::RequestContainsNul)?;
        let (return_code, response) =
            call_output(|output| self.releases.abi.prepare(&payload, output))?;
        if return_code != 0 {
            let failure = serde_json::from_str::<NativeFailure>(&response).unwrap_or_else(|_| {
                NativeFailure {
                    reason: "native_preprocess_failed".to_owned(),
                    detail: response,
                }
            });
            return Err(PreprocessError::NativeFailure {
                return_code,
                reason: failure.reason,
                detail: failure.detail,
            });
        }
        let result: NativeTensor = serde_json::from_str(&response)
            .map_err(|source| PreprocessError::InvalidTensorJson { source })?;
        self.validate_tensor_result(frame, contract, result)
    }

    fn validate_tensor_result(
        &self,
        frame: NvmmFrameInput,
        contract: TensorContract,
        result: NativeTensor,
    ) -> Result<DeviceTensor, PreprocessError> {
        let release_token = NonZeroU64::new(result.release_token)
            .ok_or_else(|| PreprocessError::invalid_result("release_token must be positive"))?;
        let invalid = if result.device_ptr == 0 {
            Some("device_ptr must be positive".to_owned())
        } else if result.shape != contract.shape.map(u64::from) {
            Some(format!(
                "shape {:?} does not match {:?}",
                result.shape,
                contract.shape()
            ))
        } else if result.dtype != contract.dtype.as_abi_str() {
            Some(format!(
                "dtype {} does not match {}",
                result.dtype,
                contract.dtype.as_abi_str()
            ))
        } else if result.nbytes != contract.nbytes {
            Some(format!(
                "nbytes {} does not match {}",
                result.nbytes, contract.nbytes
            ))
        } else if !result.zero_copy || result.memory_space != "cuda_device" {
            Some("native result is not CUDA device memory".to_owned())
        } else if result.backend != self.backend.as_ref() {
            Some(format!(
                "backend {} does not match initialized backend {}",
                result.backend, self.backend
            ))
        } else {
            None
        };
        if let Some(detail) = invalid {
            let release_code = self.releases.release_or_retain(release_token);
            return Err(PreprocessError::InvalidTensorResult {
                detail,
                release_code,
            });
        }
        Ok(DeviceTensor {
            releases: Arc::clone(&self.releases),
            release_token: Some(release_token),
            device_ptr: NonZeroU64::new(result.device_ptr)
                .expect("validated native device pointer is non-zero"),
            nbytes: result.nbytes,
            contract,
            epoch: frame.epoch,
            generation: frame.generation,
            captured_at: frame.captured_at,
            backend: result.backend,
            _thread_affine: PhantomData,
        })
    }

    /// Retries native frees retained after an earlier cleanup failure.
    /// Returns the number of allocations still pending afterwards.
    pub fn retry_pending_releases(&self) -> usize {
        self.releases.retry_pending()
    }
}

pub struct DeviceTensor {
    releases: Arc<ReleaseRegistry>,
    release_token: Option<NonZeroU64>,
    device_ptr: NonZeroU64,
    nbytes: u64,
    contract: TensorContract,
    epoch: RuntimeEpoch,
    generation: Generation,
    captured_at: MonotonicNanos,
    backend: String,
    // CUDA allocation use and release stay on the future TensorRT context
    // owner thread. The preprocessor itself remains Send + Sync.
    _thread_affine: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for DeviceTensor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceTensor")
            .field("device_ptr", &self.device_ptr)
            .field("nbytes", &self.nbytes)
            .field("contract", &self.contract)
            .field("epoch", &self.epoch)
            .field("generation", &self.generation)
            .field("captured_at", &self.captured_at)
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl DeviceTensor {
    pub const fn device_ptr(&self) -> NonZeroU64 {
        self.device_ptr
    }

    pub const fn nbytes(&self) -> u64 {
        self.nbytes
    }

    pub const fn contract(&self) -> TensorContract {
        self.contract
    }

    pub const fn epoch(&self) -> RuntimeEpoch {
        self.epoch
    }

    pub const fn generation(&self) -> Generation {
        self.generation
    }

    pub const fn captured_at(&self) -> MonotonicNanos {
        self.captured_at
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn release(&mut self) -> Result<(), PreprocessError> {
        let Some(token) = self.release_token.take() else {
            return Ok(());
        };
        let return_code = self.releases.abi.release(token.get());
        if return_code != 0 {
            self.release_token = Some(token);
            return Err(PreprocessError::Release {
                token: token.get(),
                return_code,
            });
        }
        Ok(())
    }
}

impl Drop for DeviceTensor {
    fn drop(&mut self) {
        let Some(token) = self.release_token.take() else {
            return;
        };
        self.releases.release_or_retain(token);
    }
}

struct ReleaseRegistry {
    abi: Arc<dyn NativePreprocessAbi>,
    pending: Mutex<Vec<NonZeroU64>>,
}

struct QuarantinedRelease {
    abi: Arc<dyn NativePreprocessAbi>,
    token: NonZeroU64,
}

fn quarantined_releases() -> &'static Mutex<Vec<QuarantinedRelease>> {
    static RELEASES: OnceLock<Mutex<Vec<QuarantinedRelease>>> = OnceLock::new();
    RELEASES.get_or_init(|| Mutex::new(Vec::new()))
}

fn retry_quarantined_releases() -> usize {
    let releases = {
        let mut releases = quarantined_releases()
            .lock()
            .expect("quarantined release mutex poisoned");
        std::mem::take(&mut *releases)
    };
    let mut failed = Vec::new();
    for release in releases {
        if release.abi.release(release.token.get()) != 0 {
            failed.push(release);
        }
    }
    let mut releases = quarantined_releases()
        .lock()
        .expect("quarantined release mutex poisoned");
    releases.extend(failed);
    releases.len()
}

impl ReleaseRegistry {
    #[cfg(any(feature = "ffi", test))]
    fn new(abi: Arc<dyn NativePreprocessAbi>) -> Self {
        Self {
            abi,
            pending: Mutex::new(Vec::new()),
        }
    }

    fn release_or_retain(&self, token: NonZeroU64) -> i32 {
        let return_code = self.abi.release(token.get());
        if return_code != 0 {
            self.pending
                .lock()
                .expect("release registry mutex poisoned")
                .push(token);
        }
        return_code
    }

    #[cfg(any(feature = "ffi", test))]
    fn retry_pending(&self) -> usize {
        let quarantined = retry_quarantined_releases();
        let pending = {
            let mut pending = self
                .pending
                .lock()
                .expect("release registry mutex poisoned");
            std::mem::take(&mut *pending)
        };
        let mut failed = Vec::new();
        for token in pending {
            if self.abi.release(token.get()) != 0 {
                failed.push(token);
            }
        }
        let mut pending = self
            .pending
            .lock()
            .expect("release registry mutex poisoned");
        pending.extend(failed);
        quarantined + pending.len()
    }
}

impl Drop for ReleaseRegistry {
    fn drop(&mut self) {
        retry_quarantined_releases();
        let tokens = self
            .pending
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for token in tokens.drain(..) {
            if self.abi.release(token.get()) != 0 {
                quarantined_releases()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(QuarantinedRelease {
                        abi: Arc::clone(&self.abi),
                        token,
                    });
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum PreprocessError {
    #[error("tensor dimensions must be positive")]
    ZeroTensorDimension,
    #[error("tensor byte size overflow")]
    TensorSizeOverflow,
    #[error("frame generation must be positive")]
    ZeroGeneration,
    #[error("runtime epoch must be positive")]
    ZeroEpoch,
    #[error("frame capture timestamp must be positive")]
    ZeroCaptureTimestamp,
    #[error("GstBuffer pointer must not be null")]
    NullGstBuffer,
    #[error("frame dimensions must be positive")]
    ZeroFrameDimension,
    #[error("geometry {width}x{height} exceeds native CUDA integer limits")]
    GeometryExceedsNativeLimits { width: u32, height: u32 },
    #[error("NV12 frame dimensions must be even, got {width}x{height}")]
    OddNv12Geometry { width: u32, height: u32 },
    #[error("native preprocess ABI mismatch: expected {expected}, got {actual}")]
    AbiVersion { expected: u32, actual: u32 },
    #[error("native preprocess status call failed with code {0}")]
    StatusCall(i32),
    #[error("native preprocess status is invalid JSON: {source}")]
    InvalidStatusJson { source: serde_json::Error },
    #[error("native preprocess is not ready: backend={backend} reason={reason} detail={detail}")]
    NotReady {
        backend: String,
        reason: String,
        detail: String,
    },
    #[error("native preprocess request serialization failed: {source}")]
    SerializeRequest { source: serde_json::Error },
    #[error("native preprocess request contains a NUL byte")]
    RequestContainsNul,
    #[error("native preprocess output is not NUL terminated within {OUTPUT_CAPACITY} bytes")]
    OutputTooLarge,
    #[error("native preprocess output is not UTF-8: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),
    #[error("native preprocess failed with code {return_code}: {reason}: {detail}")]
    NativeFailure {
        return_code: i32,
        reason: String,
        detail: String,
    },
    #[error("native preprocess tensor result is invalid JSON: {source}")]
    InvalidTensorJson { source: serde_json::Error },
    #[error("native preprocess tensor result is invalid: {detail}; cleanup code={release_code}")]
    InvalidTensorResult { detail: String, release_code: i32 },
    #[error("native tensor release failed for token {token} with code {return_code}")]
    Release { token: u64, return_code: i32 },
}

#[cfg(any(feature = "ffi", test))]
impl PreprocessError {
    fn invalid_result(detail: impl Into<String>) -> Self {
        Self::InvalidTensorResult {
            detail: detail.into(),
            release_code: 0,
        }
    }
}

#[cfg(any(feature = "ffi", test))]
#[derive(Debug, Deserialize)]
struct NativeStatus {
    #[serde(default)]
    available: bool,
    #[serde(default)]
    ready: bool,
    backend: String,
    #[serde(default)]
    zero_copy: bool,
    #[serde(default)]
    memory_space: String,
    abi_version: u32,
    reason: Option<String>,
    detail: Option<String>,
}

#[cfg(any(feature = "ffi", test))]
#[derive(Debug, Deserialize)]
struct NativeFailure {
    reason: String,
    #[serde(default)]
    detail: String,
}

#[cfg(any(feature = "ffi", test))]
#[derive(Debug, Deserialize)]
struct NativeTensor {
    device_ptr: u64,
    nbytes: u64,
    shape: [u64; 4],
    dtype: String,
    backend: String,
    zero_copy: bool,
    memory_space: String,
    release_token: u64,
}

#[cfg(any(feature = "ffi", test))]
#[derive(Debug, Serialize)]
struct PreparePayload {
    frame_id: u64,
    capture_ts_ns: u64,
    dmabuf_fd: i32,
    gst_buffer_ptr: u64,
    resource_kind: &'static str,
    resource_memory: &'static str,
    resource_source: &'static str,
    resource_pixel_format: &'static str,
    resource_width: u32,
    resource_height: u32,
    pixel_format: &'static str,
    width: u32,
    height: u32,
    source_width: u32,
    source_height: u32,
    roi_offset_x: u32,
    roi_offset_y: u32,
    needs_resize: bool,
    model_shape: ModelShape,
    nchw: [u32; 4],
    dtype: &'static str,
}

#[cfg(any(feature = "ffi", test))]
impl PreparePayload {
    fn new(frame: NvmmFrameInput, contract: TensorContract) -> Self {
        let width = frame.width.get();
        let height = frame.height.get();
        Self {
            frame_id: frame.generation.0,
            capture_ts_ns: frame.captured_at.0,
            dmabuf_fd: -1,
            gst_buffer_ptr: frame.gst_buffer_ptr.get(),
            resource_kind: "gstreamer_sample",
            resource_memory: "nvmm",
            resource_source: "deepstream_pad",
            resource_pixel_format: "NV12",
            resource_width: width,
            resource_height: height,
            pixel_format: "NV12",
            width,
            height,
            source_width: width,
            source_height: height,
            roi_offset_x: 0,
            roi_offset_y: 0,
            needs_resize: width != contract.shape[3] || height != contract.shape[2],
            model_shape: ModelShape {
                batch: contract.shape[0],
                channels: contract.shape[1],
                height: contract.shape[2],
                width: contract.shape[3],
            },
            nchw: contract.shape,
            dtype: contract.dtype.as_abi_str(),
        }
    }
}

#[cfg(any(feature = "ffi", test))]
#[derive(Debug, Serialize)]
struct ModelShape {
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
}

trait NativePreprocessAbi: Send + Sync {
    #[cfg(any(feature = "ffi", test))]
    fn abi_version(&self) -> u32;
    #[cfg(any(feature = "ffi", test))]
    fn status(&self, output: &mut [c_char]) -> i32;
    #[cfg(any(feature = "ffi", test))]
    fn prepare(&self, payload: &CStr, output: &mut [c_char]) -> i32;
    fn release(&self, release_token: u64) -> i32;
}

#[cfg(any(feature = "ffi", test))]
fn call_output(call: impl FnOnce(&mut [c_char]) -> i32) -> Result<(i32, String), PreprocessError> {
    let mut output = [0 as c_char; OUTPUT_CAPACITY];
    let return_code = call(&mut output);
    let end = output
        .iter()
        .position(|byte| *byte == 0)
        .ok_or(PreprocessError::OutputTooLarge)?;
    let bytes = unsafe { std::slice::from_raw_parts(output.as_ptr().cast::<u8>(), end) };
    Ok((return_code, std::str::from_utf8(bytes)?.to_owned()))
}

#[cfg(feature = "ffi")]
#[derive(Clone, Copy, Debug)]
struct LinkedNativeAbi;

#[cfg(feature = "ffi")]
impl NativePreprocessAbi for LinkedNativeAbi {
    fn abi_version(&self) -> u32 {
        unsafe { novasight_abi_version() }
    }

    fn status(&self, output: &mut [c_char]) -> i32 {
        unsafe { novasight_status_json(output.as_mut_ptr(), output.len()) }
    }

    fn prepare(&self, payload: &CStr, output: &mut [c_char]) -> i32 {
        unsafe {
            novasight_prepare_tensor_json(payload.as_ptr(), output.as_mut_ptr(), output.len())
        }
    }

    fn release(&self, release_token: u64) -> i32 {
        unsafe { novasight_release_tensor(release_token) }
    }
}

#[cfg(feature = "ffi")]
unsafe extern "C" {
    fn novasight_abi_version() -> u32;
    fn novasight_status_json(status_json: *mut c_char, status_json_size: usize) -> i32;
    fn novasight_prepare_tensor_json(
        payload_json: *const c_char,
        result_json: *mut c_char,
        result_json_size: usize,
    ) -> i32;
    fn novasight_release_tensor(release_token: u64) -> i32;
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[derive(Debug)]
    struct FakeAbi {
        version: u32,
        status: String,
        prepare_code: i32,
        result: String,
        payloads: Mutex<Vec<serde_json::Value>>,
        releases: Mutex<Vec<u64>>,
        release_codes: Mutex<Vec<i32>>,
    }

    impl FakeAbi {
        fn ready(result: impl Into<String>) -> Arc<Self> {
            Arc::new(Self {
                version: ABI_VERSION,
                status: r#"{"available":true,"ready":true,"backend":"fake_cuda","zero_copy":true,"memory_space":"cuda_device","abi_version":1}"#.to_owned(),
                prepare_code: 0,
                result: result.into(),
                payloads: Mutex::new(Vec::new()),
                releases: Mutex::new(Vec::new()),
                release_codes: Mutex::new(Vec::new()),
            })
        }
    }

    impl NativePreprocessAbi for FakeAbi {
        fn abi_version(&self) -> u32 {
            self.version
        }

        fn status(&self, output: &mut [c_char]) -> i32 {
            write_output(output, &self.status);
            0
        }

        fn prepare(&self, payload: &CStr, output: &mut [c_char]) -> i32 {
            self.payloads
                .lock()
                .unwrap()
                .push(serde_json::from_slice(payload.to_bytes()).expect("typed request JSON"));
            write_output(output, &self.result);
            self.prepare_code
        }

        fn release(&self, release_token: u64) -> i32 {
            self.releases.lock().unwrap().push(release_token);
            let mut codes = self.release_codes.lock().unwrap();
            if codes.is_empty() { 0 } else { codes.remove(0) }
        }
    }

    fn write_output(output: &mut [c_char], value: &str) {
        assert!(value.len() < output.len());
        for (target, source) in output.iter_mut().zip(value.bytes()) {
            *target = source as c_char;
        }
        output[value.len()] = 0;
    }

    fn frame() -> NvmmFrameInput {
        NvmmFrameInput::from_deepstream_pad(
            RuntimeEpoch(7),
            Generation(11),
            MonotonicNanos(123_456),
            0x1234,
            640,
            640,
        )
        .unwrap()
    }

    fn contract() -> TensorContract {
        TensorContract::rgb_nchw(640, 640, TensorDtype::Float16).unwrap()
    }

    fn valid_result() -> String {
        format!(
            r#"{{"device_ptr":4096,"nbytes":{},"shape":[1,3,640,640],"dtype":"float16","backend":"fake_cuda","zero_copy":true,"memory_space":"cuda_device","release_token":41}}"#,
            contract().nbytes()
        )
    }

    #[test]
    fn prepare_hides_native_payload_and_preserves_frame_identity() {
        let abi = FakeAbi::ready(valid_result());
        let preprocessor = CudaPreprocessor::from_abi(abi.clone()).unwrap();

        let mut tensor = preprocessor.prepare(frame(), contract()).unwrap();

        assert_eq!(tensor.device_ptr(), NonZeroU64::new(4096).unwrap());
        assert_eq!(tensor.contract(), contract());
        assert_eq!(tensor.epoch(), RuntimeEpoch(7));
        assert_eq!(tensor.generation(), Generation(11));
        assert_eq!(tensor.captured_at(), MonotonicNanos(123_456));
        let payloads = abi.payloads.lock().unwrap();
        assert_eq!(payloads[0]["resource_source"], "deepstream_pad");
        assert_eq!(payloads[0]["gst_buffer_ptr"], 0x1234);
        assert_eq!(payloads[0]["nchw"], serde_json::json!([1, 3, 640, 640]));
        drop(payloads);

        tensor.release().unwrap();
        drop(tensor);
        assert_eq!(*abi.releases.lock().unwrap(), vec![41]);
    }

    #[test]
    fn tensor_drop_releases_native_allocation_exactly_once() {
        let abi = FakeAbi::ready(valid_result());
        let preprocessor = CudaPreprocessor::from_abi(abi.clone()).unwrap();

        drop(preprocessor.prepare(frame(), contract()).unwrap());

        assert_eq!(*abi.releases.lock().unwrap(), vec![41]);
    }

    #[test]
    fn invalid_success_result_is_released_before_rejection() {
        let abi = FakeAbi::ready(
            r#"{"device_ptr":4096,"nbytes":1,"shape":[1,3,640,640],"dtype":"float16","backend":"fake_cuda","zero_copy":true,"memory_space":"cuda_device","release_token":99}"#,
        );
        let preprocessor = CudaPreprocessor::from_abi(abi.clone()).unwrap();

        let error = preprocessor.prepare(frame(), contract()).unwrap_err();

        assert!(matches!(error, PreprocessError::InvalidTensorResult { .. }));
        assert_eq!(*abi.releases.lock().unwrap(), vec![99]);
    }

    #[test]
    fn invalid_result_cleanup_failure_retains_token_until_retry_succeeds() {
        let abi = FakeAbi::ready(
            r#"{"device_ptr":4096,"nbytes":1,"shape":[1,3,640,640],"dtype":"float16","backend":"fake_cuda","zero_copy":true,"memory_space":"cuda_device","release_token":99}"#,
        );
        *abi.release_codes.lock().unwrap() = vec![7, 7, 0];
        let preprocessor = CudaPreprocessor::from_abi(abi.clone()).unwrap();

        let error = preprocessor.prepare(frame(), contract()).unwrap_err();

        assert!(matches!(
            error,
            PreprocessError::InvalidTensorResult {
                release_code: 7,
                ..
            }
        ));
        assert_eq!(preprocessor.retry_pending_releases(), 1);
        assert_eq!(preprocessor.retry_pending_releases(), 0);
        assert_eq!(*abi.releases.lock().unwrap(), vec![99, 99, 99]);
    }

    #[test]
    fn drop_cleanup_failure_is_owned_by_preprocessor_retry_queue() {
        let abi = FakeAbi::ready(valid_result());
        *abi.release_codes.lock().unwrap() = vec![7, 7, 0];
        let preprocessor = CudaPreprocessor::from_abi(abi.clone()).unwrap();

        drop(preprocessor.prepare(frame(), contract()).unwrap());

        assert_eq!(preprocessor.retry_pending_releases(), 1);
        assert_eq!(preprocessor.retry_pending_releases(), 0);
        assert_eq!(*abi.releases.lock().unwrap(), vec![41, 41, 41]);
    }

    #[test]
    fn pending_release_survives_preprocessor_drop_in_process_quarantine() {
        let abi = FakeAbi::ready(valid_result());
        *abi.release_codes.lock().unwrap() = vec![7, 7, 0];
        let preprocessor = CudaPreprocessor::from_abi(abi.clone()).unwrap();
        drop(preprocessor.prepare(frame(), contract()).unwrap());

        drop(preprocessor);
        retry_quarantined_releases();

        assert_eq!(*abi.releases.lock().unwrap(), vec![41, 41, 41]);
    }

    #[test]
    fn unavailable_native_backend_fails_closed_before_prepare() {
        let abi = Arc::new(FakeAbi {
            version: ABI_VERSION,
            status: r#"{"available":false,"ready":false,"backend":"reference","zero_copy":false,"memory_space":"","abi_version":1,"reason":"not_jetson"}"#.to_owned(),
            prepare_code: 0,
            result: valid_result(),
            payloads: Mutex::new(Vec::new()),
            releases: Mutex::new(Vec::new()),
            release_codes: Mutex::new(Vec::new()),
        });

        let error = CudaPreprocessor::from_abi(abi.clone()).unwrap_err();

        assert!(matches!(error, PreprocessError::NotReady { .. }));
        assert!(abi.payloads.lock().unwrap().is_empty());
    }

    #[test]
    fn tensor_contract_rejects_zero_and_native_unsafe_geometry() {
        assert!(matches!(
            TensorContract::rgb_nchw(0, 640, TensorDtype::Float16),
            Err(PreprocessError::ZeroTensorDimension)
        ));
        assert!(matches!(
            TensorContract::rgb_nchw(u32::MAX, u32::MAX, TensorDtype::Float32),
            Err(PreprocessError::GeometryExceedsNativeLimits { .. })
        ));
        assert!(matches!(
            NvmmFrameInput::from_deepstream_pad(
                RuntimeEpoch(0),
                Generation(1),
                MonotonicNanos(1),
                1,
                1,
                1,
            ),
            Err(PreprocessError::ZeroEpoch)
        ));
        assert!(matches!(
            NvmmFrameInput::from_deepstream_pad(
                RuntimeEpoch(1),
                Generation(1),
                MonotonicNanos(1),
                1,
                641,
                640,
            ),
            Err(PreprocessError::OddNv12Geometry { .. })
        ));
        assert!(matches!(
            NvmmFrameInput::from_deepstream_pad(
                RuntimeEpoch(1),
                Generation(1),
                MonotonicNanos(1),
                1,
                (i32::MAX as u32) + 1,
                640,
            ),
            Err(PreprocessError::GeometryExceedsNativeLimits { .. })
        ));
    }

    #[test]
    fn failed_release_retains_token_for_drop_retry() {
        let abi = FakeAbi::ready(valid_result());
        abi.release_codes.lock().unwrap().extend([7, 0]);
        let preprocessor = CudaPreprocessor::from_abi(abi.clone()).unwrap();
        let mut tensor = preprocessor.prepare(frame(), contract()).unwrap();

        assert!(matches!(
            tensor.release(),
            Err(PreprocessError::Release {
                token: 41,
                return_code: 7,
            })
        ));
        drop(tensor);

        assert_eq!(*abi.releases.lock().unwrap(), vec![41, 41]);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn linked_library_exposes_a_valid_versioned_status_contract() {
        match CudaPreprocessor::linked() {
            Ok(preprocessor) => assert!(!preprocessor.backend().trim().is_empty()),
            Err(PreprocessError::NotReady { backend, .. }) => {
                assert!(!backend.trim().is_empty())
            }
            Err(other) => panic!("linked native status contract failed: {other}"),
        }
    }
}
