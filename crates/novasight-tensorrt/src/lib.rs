#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(not(any(feature = "ffi", test)), allow(dead_code))]

use std::ffi::{CStr, CString, c_char, c_void};
use std::marker::PhantomData;
use std::num::{NonZeroU32, NonZeroU64};
use std::path::Path;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

use novasight_core::{Generation, MonotonicNanos, RuntimeEpoch};
use thiserror::Error;

#[cfg_attr(
    not(all(feature = "gpu-frame", target_os = "linux", target_arch = "aarch64")),
    allow(dead_code)
)]
pub mod gpu;

mod decoder;
pub use decoder::{DecodeContract, DecodeError, DetectionDecoder};

const ABI_VERSION: u32 = 3;
const MAX_NAME: usize = 128;
const MAX_DEVICE_NAME: usize = 256;
const MAX_RANK: usize = 8;
const MAX_OUTPUTS: usize = 8;
const ERROR_CAPACITY: usize = 1024;
const MAX_NATIVE_DIMENSION: u32 = i32::MAX as u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorDtype {
    Float16,
    Float32,
}

impl TensorDtype {
    const fn byte_width(self) -> u64 {
        match self {
            Self::Float16 => 2,
            Self::Float32 => 4,
        }
    }
}

/// Batch-one RGB NCHW tensor contract used when inspecting an Engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorContract {
    shape: [u32; 4],
    dtype: TensorDtype,
    nbytes: u64,
}

impl TensorContract {
    pub fn rgb_nchw(
        height: u32,
        width: u32,
        dtype: TensorDtype,
    ) -> Result<Self, TensorContractError> {
        let height = NonZeroU32::new(height).ok_or(TensorContractError::ZeroDimension)?;
        let width = NonZeroU32::new(width).ok_or(TensorContractError::ZeroDimension)?;
        if height.get() > MAX_NATIVE_DIMENSION || width.get() > MAX_NATIVE_DIMENSION {
            return Err(TensorContractError::DimensionTooLarge {
                width: width.get(),
                height: height.get(),
            });
        }
        let nbytes = 3_u64
            .checked_mul(u64::from(height.get()))
            .and_then(|value| value.checked_mul(u64::from(width.get())))
            .and_then(|value| value.checked_mul(dtype.byte_width()))
            .ok_or(TensorContractError::SizeOverflow)?;
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

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum TensorContractError {
    #[error("tensor dimensions must be positive")]
    ZeroDimension,
    #[error("tensor dimensions exceed native limits: {width}x{height}")]
    DimensionTooLarge { width: u32, height: u32 },
    #[error("tensor byte size overflow")]
    SizeOverflow,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSpec {
    name: String,
    dimensions: [u64; MAX_RANK],
    rank: usize,
    dtype: TensorDtype,
    nbytes: u64,
}

impl TensorSpec {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions[..self.rank]
    }

    pub const fn dtype(&self) -> TensorDtype {
        self.dtype
    }

    pub const fn nbytes(&self) -> u64 {
        self.nbytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineContract {
    input: TensorSpec,
    outputs: Vec<TensorSpec>,
    input_dynamic: bool,
    selected_profile: u32,
}

/// Runtime identity that determines whether a previous Engine validation
/// receipt still belongs to the current Jetson environment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorRtEnvironment {
    runtime_abi_version: u32,
    tensorrt_runtime_version: i32,
    cuda_runtime_version: i32,
    cuda_driver_version: i32,
    device_ordinal: i32,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    integrated: bool,
    total_global_memory: u64,
    device_name: String,
}

impl TensorRtEnvironment {
    pub const fn runtime_abi_version(&self) -> u32 {
        self.runtime_abi_version
    }

    pub const fn tensorrt_runtime_version(&self) -> i32 {
        self.tensorrt_runtime_version
    }

    pub const fn cuda_runtime_version(&self) -> i32 {
        self.cuda_runtime_version
    }

    pub const fn cuda_driver_version(&self) -> i32 {
        self.cuda_driver_version
    }

    pub const fn device_ordinal(&self) -> i32 {
        self.device_ordinal
    }

    pub const fn compute_capability_major(&self) -> i32 {
        self.compute_capability_major
    }

    pub const fn compute_capability_minor(&self) -> i32 {
        self.compute_capability_minor
    }

    pub const fn integrated(&self) -> bool {
        self.integrated
    }

    pub const fn total_global_memory(&self) -> u64 {
        self.total_global_memory
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}

impl EngineContract {
    pub const fn input(&self) -> &TensorSpec {
        &self.input
    }

    pub fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    pub const fn input_dynamic(&self) -> bool {
        self.input_dynamic
    }

    pub const fn selected_profile(&self) -> u32 {
        self.selected_profile
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A live CUDA allocation accepted by the synchronous TensorRT owner.
///
/// This trait is sealed: safe downstream code cannot manufacture a device
/// pointer or weaken the allocation-lifetime guarantee.
pub trait CudaTensorInput: sealed::Sealed {
    fn device_ptr(&self) -> NonZeroU64;
    fn nbytes(&self) -> u64;
    fn contract(&self) -> TensorContract;
    fn epoch(&self) -> RuntimeEpoch;
    fn generation(&self) -> Generation;
    fn captured_at(&self) -> MonotonicNanos;
}

pub struct TensorRtEngine {
    abi: Arc<dyn NativeTensorRtAbi>,
    handle: NonNull<c_void>,
    contract: EngineContract,
    input_contract: TensorContract,
    _thread_affine: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for TensorRtEngine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorRtEngine")
            .field("contract", &self.contract)
            .finish_non_exhaustive()
    }
}

impl TensorRtEngine {
    #[cfg(feature = "ffi")]
    pub fn preflight() -> Result<(), TensorRtError> {
        let actual = LinkedTensorRtAbi.abi_version();
        if actual != ABI_VERSION {
            return Err(TensorRtError::AbiVersion {
                expected: ABI_VERSION,
                actual,
            });
        }
        Ok(())
    }

    #[cfg(feature = "ffi")]
    pub fn environment() -> Result<TensorRtEnvironment, TensorRtError> {
        Self::environment_from_abi(&LinkedTensorRtAbi)
    }

    fn environment_from_abi(
        abi: &dyn NativeTensorRtAbi,
    ) -> Result<TensorRtEnvironment, TensorRtError> {
        let actual = abi.abi_version();
        if actual != ABI_VERSION {
            return Err(TensorRtError::AbiVersion {
                expected: ABI_VERSION,
                actual,
            });
        }
        let mut native = NativeTensorRtEnvironment::empty();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let code = unsafe { abi.environment(&mut native, &mut error) };
        if code != 0 {
            return Err(TensorRtError::NativeEnvironment {
                code,
                detail: error_text(&error),
            });
        }
        if native.runtime_abi_version != ABI_VERSION {
            return Err(TensorRtError::AbiVersion {
                expected: ABI_VERSION,
                actual: native.runtime_abi_version,
            });
        }
        let integrated = match native.integrated {
            0 => false,
            1 => true,
            value => return Err(TensorRtError::InvalidIntegratedFlag(value)),
        };
        Ok(TensorRtEnvironment {
            runtime_abi_version: native.runtime_abi_version,
            tensorrt_runtime_version: native.tensorrt_runtime_version,
            cuda_runtime_version: native.cuda_runtime_version,
            cuda_driver_version: native.cuda_driver_version,
            device_ordinal: native.device_ordinal,
            compute_capability_major: native.compute_capability_major,
            compute_capability_minor: native.compute_capability_minor,
            integrated,
            total_global_memory: native.total_global_memory,
            device_name: native_string(&native.device_name)?.to_owned(),
        })
    }

    #[cfg(feature = "ffi")]
    pub fn load(path: &Path, input: TensorContract) -> Result<Self, TensorRtError> {
        Self::from_abi(Arc::new(LinkedTensorRtAbi), path, input)
    }

    /// Load an Engine using its static input shape or optimization profile 0
    /// OPT shape. Live inference still supplies an explicit input contract;
    /// this automatic path exists only for offline inspection and probing.
    #[cfg(feature = "ffi")]
    pub fn inspect(path: &Path) -> Result<Self, TensorRtError> {
        Self::inspect_from_abi(Arc::new(LinkedTensorRtAbi), path)
    }

    fn inspect_from_abi(
        abi: Arc<dyn NativeTensorRtAbi>,
        path: &Path,
    ) -> Result<Self, TensorRtError> {
        let actual = abi.abi_version();
        if actual != ABI_VERSION {
            return Err(TensorRtError::AbiVersion {
                expected: ABI_VERSION,
                actual,
            });
        }
        let path = path
            .to_str()
            .ok_or_else(|| TensorRtError::NonUtf8Path(path.to_path_buf()))?;
        let path = CString::new(path).map_err(|_| TensorRtError::PathContainsNul)?;
        let mut handle = std::ptr::null_mut();
        let mut native_spec = NativeEngineSpec::empty();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let code = unsafe { abi.create_auto(&path, &mut handle, &mut native_spec, &mut error) };
        if code != 0 {
            return Err(TensorRtError::NativeCreate {
                code,
                detail: error_text(&error),
            });
        }
        let handle = NonNull::new(handle).ok_or(TensorRtError::NullEngineHandle)?;
        let contract = match decode_engine_contract(&native_spec) {
            Ok(contract) => contract,
            Err(error) => {
                unsafe { abi.destroy(handle.as_ptr()) };
                return Err(error);
            }
        };
        let dimensions = contract.input.dimensions();
        if dimensions.len() != 4 {
            unsafe { abi.destroy(handle.as_ptr()) };
            return Err(TensorRtError::InspectedInput(
                "input must be rank-4 NCHW".to_owned(),
            ));
        }
        // Native TensorRT dimensions are positive `int`, so these conversions
        // are lossless on every supported target.
        let input_contract = match TensorContract::rgb_nchw(
            dimensions[2] as u32,
            dimensions[3] as u32,
            contract.input.dtype(),
        ) {
            Ok(contract) => contract,
            Err(error) => {
                unsafe { abi.destroy(handle.as_ptr()) };
                return Err(TensorRtError::InspectedInput(error.to_string()));
            }
        };
        if !input_matches(&contract.input, input_contract) {
            unsafe { abi.destroy(handle.as_ptr()) };
            return Err(TensorRtError::InputContractMismatch {
                expected: input_contract,
                actual: Box::new(contract.input),
            });
        }
        Ok(Self {
            abi,
            handle,
            contract,
            input_contract,
            _thread_affine: PhantomData,
        })
    }

    fn from_abi(
        abi: Arc<dyn NativeTensorRtAbi>,
        path: &Path,
        input: TensorContract,
    ) -> Result<Self, TensorRtError> {
        let actual = abi.abi_version();
        if actual != ABI_VERSION {
            return Err(TensorRtError::AbiVersion {
                expected: ABI_VERSION,
                actual,
            });
        }
        let path = path
            .to_str()
            .ok_or_else(|| TensorRtError::NonUtf8Path(path.to_path_buf()))?;
        let path = CString::new(path).map_err(|_| TensorRtError::PathContainsNul)?;
        let shape = input.shape().map(u64::from);
        let mut handle = std::ptr::null_mut();
        let mut native_spec = NativeEngineSpec::empty();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let code = unsafe { abi.create(&path, &shape, &mut handle, &mut native_spec, &mut error) };
        if code != 0 {
            return Err(TensorRtError::NativeCreate {
                code,
                detail: error_text(&error),
            });
        }
        let handle = NonNull::new(handle).ok_or(TensorRtError::NullEngineHandle)?;
        let contract = match decode_engine_contract(&native_spec) {
            Ok(contract) => contract,
            Err(error) => {
                unsafe { abi.destroy(handle.as_ptr()) };
                return Err(error);
            }
        };
        if !input_matches(&contract.input, input) {
            unsafe { abi.destroy(handle.as_ptr()) };
            return Err(TensorRtError::InputContractMismatch {
                expected: input,
                actual: Box::new(contract.input),
            });
        }
        Ok(Self {
            abi,
            handle,
            contract,
            input_contract: input,
            _thread_affine: PhantomData,
        })
    }

    pub const fn contract(&self) -> &EngineContract {
        &self.contract
    }

    pub fn execute<'engine>(
        &'engine mut self,
        input: &impl CudaTensorInput,
    ) -> Result<ExecutionOutputs<'engine>, TensorRtError> {
        if input.contract() != self.input_contract || input.nbytes() != self.input_contract.nbytes()
        {
            return Err(TensorRtError::DeviceInputContractMismatch);
        }
        let mut dimensions = [0_u64; MAX_RANK];
        for (target, source) in dimensions.iter_mut().zip(input.contract().shape()) {
            *target = u64::from(source);
        }
        let native_input = NativeDeviceTensorView {
            device_ptr: input.device_ptr().get(),
            nbytes: input.nbytes(),
            rank: 4,
            dimensions,
            dtype: dtype_code(input.contract().dtype()),
        };
        let mut views = [NativeHostTensorView::empty(); MAX_OUTPUTS];
        let mut output_count = 0_u32;
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let code = unsafe {
            self.abi.execute(
                self.handle.as_ptr(),
                &native_input,
                &mut views,
                &mut output_count,
                &mut error,
            )
        };
        if code != 0 {
            return Err(TensorRtError::NativeExecute {
                code,
                detail: error_text(&error),
            });
        }
        let count =
            usize::try_from(output_count).map_err(|_| TensorRtError::OutputCount(output_count))?;
        if count != self.contract.outputs.len() || count > MAX_OUTPUTS {
            return Err(TensorRtError::OutputCount(output_count));
        }
        for (index, view) in views[..count].iter().enumerate() {
            let actual = decode_tensor_spec(&view.spec)?;
            if actual != self.contract.outputs[index]
                || view.nbytes != actual.nbytes
                || view.host_ptr.is_null()
                || view.nbytes > usize::MAX as u64
            {
                return Err(TensorRtError::InvalidOutputView { index });
            }
        }
        Ok(ExecutionOutputs {
            views,
            count,
            epoch: input.epoch(),
            generation: input.generation(),
            captured_at: input.captured_at(),
            _engine: PhantomData,
        })
    }

    /// Execute one deterministic all-zero input for offline model admission.
    /// The native runtime owns the temporary CUDA allocation; the returned
    /// tensors pass through the same Rust decoder as live inference.
    pub fn probe_zero(&mut self) -> Result<ExecutionOutputs<'_>, TensorRtError> {
        let mut views = [NativeHostTensorView::empty(); MAX_OUTPUTS];
        let mut output_count = 0_u32;
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let code = unsafe {
            self.abi.probe_zero(
                self.handle.as_ptr(),
                &mut views,
                &mut output_count,
                &mut error,
            )
        };
        if code != 0 {
            return Err(TensorRtError::NativeProbe {
                code,
                detail: error_text(&error),
            });
        }
        let count = validate_output_views(&self.contract, &views, output_count)?;
        Ok(ExecutionOutputs {
            views,
            count,
            epoch: RuntimeEpoch(1),
            generation: Generation(1),
            captured_at: MonotonicNanos(1),
            _engine: PhantomData,
        })
    }
}

fn validate_output_views(
    contract: &EngineContract,
    views: &[NativeHostTensorView; MAX_OUTPUTS],
    output_count: u32,
) -> Result<usize, TensorRtError> {
    let count =
        usize::try_from(output_count).map_err(|_| TensorRtError::OutputCount(output_count))?;
    if count != contract.outputs.len() || count > MAX_OUTPUTS {
        return Err(TensorRtError::OutputCount(output_count));
    }
    for (index, view) in views[..count].iter().enumerate() {
        let actual = decode_tensor_spec(&view.spec)?;
        if actual != contract.outputs[index]
            || view.nbytes != actual.nbytes
            || view.host_ptr.is_null()
            || view.nbytes > usize::MAX as u64
        {
            return Err(TensorRtError::InvalidOutputView { index });
        }
    }
    Ok(count)
}

impl Drop for TensorRtEngine {
    fn drop(&mut self) {
        unsafe { self.abi.destroy(self.handle.as_ptr()) };
    }
}

pub struct ExecutionOutputs<'engine> {
    views: [NativeHostTensorView; MAX_OUTPUTS],
    count: usize,
    epoch: RuntimeEpoch,
    generation: Generation,
    captured_at: MonotonicNanos,
    _engine: PhantomData<&'engine mut TensorRtEngine>,
}

impl ExecutionOutputs<'_> {
    pub const fn epoch(&self) -> RuntimeEpoch {
        self.epoch
    }

    pub const fn generation(&self) -> Generation {
        self.generation
    }

    pub const fn captured_at(&self) -> MonotonicNanos {
        self.captured_at
    }

    pub const fn len(&self) -> usize {
        self.count
    }

    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = HostTensor<'_>> {
        self.views[..self.count]
            .iter()
            .map(|view| HostTensor { view })
    }

    pub fn find(&self, name: &str) -> Result<HostTensor<'_>, TensorRtError> {
        self.iter()
            .find(|tensor| tensor.name().is_ok_and(|actual| actual == name))
            .ok_or_else(|| TensorRtError::MissingOutput(name.to_owned()))
    }
}

#[derive(Clone, Copy)]
pub struct HostTensor<'output> {
    view: &'output NativeHostTensorView,
}

impl HostTensor<'_> {
    pub fn name(&self) -> Result<&str, TensorRtError> {
        native_name(&self.view.spec.name)
    }

    pub fn dimensions(&self) -> &[i64] {
        &self.view.spec.dimensions[..self.view.spec.rank as usize]
    }

    pub fn dtype(&self) -> Result<TensorDtype, TensorRtError> {
        decode_dtype(self.view.spec.dtype)
    }

    pub fn bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.view.host_ptr.cast::<u8>(), self.view.nbytes as usize)
        }
    }

    pub fn element_count(&self) -> usize {
        match self.view.spec.dtype {
            1 => self.view.nbytes as usize / 4,
            2 => self.view.nbytes as usize / 2,
            _ => 0,
        }
    }

    pub fn value_f32(&self, index: usize) -> Result<f32, TensorRtError> {
        if index >= self.element_count() {
            return Err(TensorRtError::OutputIndex {
                index,
                elements: self.element_count(),
            });
        }
        let bytes = self.bytes();
        match self.dtype()? {
            TensorDtype::Float32 => {
                let offset = index * 4;
                Ok(f32::from_ne_bytes(
                    bytes[offset..offset + 4]
                        .try_into()
                        .expect("validated f32 element width"),
                ))
            }
            TensorDtype::Float16 => {
                let offset = index * 2;
                let bits = u16::from_ne_bytes(
                    bytes[offset..offset + 2]
                        .try_into()
                        .expect("validated f16 element width"),
                );
                Ok(f16_bits_to_f32(bits))
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum TensorRtError {
    #[error("TensorRT runtime ABI mismatch: expected {expected}, got {actual}")]
    AbiVersion { expected: u32, actual: u32 },
    #[error("TensorRT environment query failed with code {code}: {detail}")]
    NativeEnvironment { code: i32, detail: String },
    #[error("TensorRT returned invalid integrated-device flag {0}")]
    InvalidIntegratedFlag(u32),
    #[error("TensorRT engine path is not UTF-8: {}", .0.display())]
    NonUtf8Path(std::path::PathBuf),
    #[error("TensorRT engine path contains a NUL byte")]
    PathContainsNul,
    #[error("TensorRT create failed with code {code}: {detail}")]
    NativeCreate { code: i32, detail: String },
    #[error("TensorRT create returned a null engine handle")]
    NullEngineHandle,
    #[error("TensorRT input contract mismatch: expected {expected:?}, got {actual:?}")]
    InputContractMismatch {
        expected: TensorContract,
        actual: Box<TensorSpec>,
    },
    #[error("CUDA device input does not match the loaded TensorRT input contract")]
    DeviceInputContractMismatch,
    #[error("TensorRT execute failed with code {code}: {detail}")]
    NativeExecute { code: i32, detail: String },
    #[error("TensorRT zero probe failed with code {code}: {detail}")]
    NativeProbe { code: i32, detail: String },
    #[error("TensorRT inspected input is unsupported: {0}")]
    InspectedInput(String),
    #[error("TensorRT returned invalid output count {0}")]
    OutputCount(u32),
    #[error("TensorRT returned invalid output view at index {index}")]
    InvalidOutputView { index: usize },
    #[error("TensorRT tensor name is not NUL terminated")]
    UnterminatedName,
    #[error("TensorRT tensor name is not UTF-8")]
    InvalidNameUtf8,
    #[error("TensorRT tensor rank {0} is invalid")]
    InvalidRank(u32),
    #[error("TensorRT tensor dimension {0} is unresolved or invalid")]
    InvalidDimension(i64),
    #[error("TensorRT tensor dtype code {0} is unsupported")]
    InvalidDtype(u32),
    #[error("TensorRT tensor byte size overflow")]
    TensorSizeOverflow,
    #[error("TensorRT tensor nbytes {actual} does not match shape-derived {expected}")]
    TensorNbytes { expected: u64, actual: u64 },
    #[error("TensorRT engine must expose between 1 and {MAX_OUTPUTS} outputs, got {0}")]
    InvalidEngineOutputCount(u32),
    #[error("TensorRT engine exposes duplicate output name {0}")]
    DuplicateOutputName(String),
    #[error("TensorRT returned invalid dynamic-input flag {0}")]
    InvalidDynamicFlag(u32),
    #[error("TensorRT output {0} is missing")]
    MissingOutput(String),
    #[error("TensorRT output index {index} exceeds {elements} elements")]
    OutputIndex { index: usize, elements: usize },
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let expanded = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            let normalized = fraction << shift;
            let exponent32 = 127_u32 - 14 - shift;
            sign | (exponent32 << 23) | ((normalized & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(expanded)
}

fn input_matches(spec: &TensorSpec, contract: TensorContract) -> bool {
    spec.dimensions() == contract.shape().map(u64::from)
        && spec.dtype == contract.dtype()
        && spec.nbytes == contract.nbytes()
}

fn decode_engine_contract(native: &NativeEngineSpec) -> Result<EngineContract, TensorRtError> {
    let input = decode_tensor_spec(&native.input)?;
    let output_count = usize::try_from(native.output_count)
        .map_err(|_| TensorRtError::InvalidEngineOutputCount(native.output_count))?;
    if output_count == 0 || output_count > MAX_OUTPUTS {
        return Err(TensorRtError::InvalidEngineOutputCount(native.output_count));
    }
    let mut outputs = Vec::with_capacity(output_count);
    for native_output in &native.outputs[..output_count] {
        let output = decode_tensor_spec(native_output)?;
        if outputs
            .iter()
            .any(|existing: &TensorSpec| existing.name == output.name)
        {
            return Err(TensorRtError::DuplicateOutputName(output.name));
        }
        outputs.push(output);
    }
    if native.input_dynamic > 1 {
        return Err(TensorRtError::InvalidDynamicFlag(native.input_dynamic));
    }
    Ok(EngineContract {
        input,
        outputs,
        input_dynamic: native.input_dynamic == 1,
        selected_profile: native.selected_profile,
    })
}

fn decode_tensor_spec(native: &NativeTensorSpec) -> Result<TensorSpec, TensorRtError> {
    let rank = usize::try_from(native.rank).map_err(|_| TensorRtError::InvalidRank(native.rank))?;
    if rank == 0 || rank > MAX_RANK {
        return Err(TensorRtError::InvalidRank(native.rank));
    }
    let mut dimensions = [0_u64; MAX_RANK];
    let mut elements = 1_u64;
    for (index, dimension) in native.dimensions[..rank].iter().copied().enumerate() {
        if dimension <= 0 {
            return Err(TensorRtError::InvalidDimension(dimension));
        }
        let dimension =
            u64::try_from(dimension).map_err(|_| TensorRtError::InvalidDimension(dimension))?;
        elements = elements
            .checked_mul(dimension)
            .ok_or(TensorRtError::TensorSizeOverflow)?;
        dimensions[index] = dimension;
    }
    let dtype = decode_dtype(native.dtype)?;
    let expected = elements
        .checked_mul(match dtype {
            TensorDtype::Float16 => 2,
            TensorDtype::Float32 => 4,
        })
        .ok_or(TensorRtError::TensorSizeOverflow)?;
    if expected != native.nbytes {
        return Err(TensorRtError::TensorNbytes {
            expected,
            actual: native.nbytes,
        });
    }
    Ok(TensorSpec {
        name: native_name(&native.name)?.to_owned(),
        dimensions,
        rank,
        dtype,
        nbytes: native.nbytes,
    })
}

fn native_name(name: &[c_char; MAX_NAME]) -> Result<&str, TensorRtError> {
    native_string(name)
}

fn native_string<const N: usize>(name: &[c_char; N]) -> Result<&str, TensorRtError> {
    let end = name
        .iter()
        .position(|value| *value == 0)
        .ok_or(TensorRtError::UnterminatedName)?;
    let bytes = unsafe { std::slice::from_raw_parts(name.as_ptr().cast::<u8>(), end) };
    std::str::from_utf8(bytes).map_err(|_| TensorRtError::InvalidNameUtf8)
}

fn dtype_code(dtype: TensorDtype) -> u32 {
    match dtype {
        TensorDtype::Float32 => 1,
        TensorDtype::Float16 => 2,
    }
}

fn decode_dtype(code: u32) -> Result<TensorDtype, TensorRtError> {
    match code {
        1 => Ok(TensorDtype::Float32),
        2 => Ok(TensorDtype::Float16),
        other => Err(TensorRtError::InvalidDtype(other)),
    }
}

fn error_text(error: &[c_char]) -> String {
    let end = error
        .iter()
        .position(|value| *value == 0)
        .unwrap_or(error.len());
    let bytes = unsafe { std::slice::from_raw_parts(error.as_ptr().cast::<u8>(), end) };
    String::from_utf8_lossy(bytes).into_owned()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeTensorSpec {
    name: [c_char; MAX_NAME],
    rank: u32,
    dimensions: [i64; MAX_RANK],
    dtype: u32,
    nbytes: u64,
}

impl NativeTensorSpec {
    const fn empty() -> Self {
        Self {
            name: [0; MAX_NAME],
            rank: 0,
            dimensions: [0; MAX_RANK],
            dtype: 0,
            nbytes: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeEngineSpec {
    input: NativeTensorSpec,
    output_count: u32,
    outputs: [NativeTensorSpec; MAX_OUTPUTS],
    input_dynamic: u32,
    selected_profile: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeTensorRtEnvironment {
    runtime_abi_version: u32,
    tensorrt_runtime_version: i32,
    cuda_runtime_version: i32,
    cuda_driver_version: i32,
    device_ordinal: i32,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    integrated: u32,
    total_global_memory: u64,
    device_name: [c_char; MAX_DEVICE_NAME],
}

impl NativeTensorRtEnvironment {
    const fn empty() -> Self {
        Self {
            runtime_abi_version: 0,
            tensorrt_runtime_version: 0,
            cuda_runtime_version: 0,
            cuda_driver_version: 0,
            device_ordinal: 0,
            compute_capability_major: 0,
            compute_capability_minor: 0,
            integrated: 0,
            total_global_memory: 0,
            device_name: [0; MAX_DEVICE_NAME],
        }
    }
}

impl NativeEngineSpec {
    const fn empty() -> Self {
        Self {
            input: NativeTensorSpec::empty(),
            output_count: 0,
            outputs: [NativeTensorSpec::empty(); MAX_OUTPUTS],
            input_dynamic: 0,
            selected_profile: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeDeviceTensorView {
    device_ptr: u64,
    nbytes: u64,
    rank: u32,
    dimensions: [u64; MAX_RANK],
    dtype: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeHostTensorView {
    host_ptr: *const c_void,
    nbytes: u64,
    spec: NativeTensorSpec,
}

impl NativeHostTensorView {
    const fn empty() -> Self {
        Self {
            host_ptr: std::ptr::null(),
            nbytes: 0,
            spec: NativeTensorSpec::empty(),
        }
    }
}

trait NativeTensorRtAbi: Send + Sync {
    fn abi_version(&self) -> u32;

    unsafe fn environment(
        &self,
        environment: &mut NativeTensorRtEnvironment,
        error: &mut [c_char],
    ) -> i32;

    unsafe fn create(
        &self,
        path: &CStr,
        input_shape: &[u64; 4],
        handle: &mut *mut c_void,
        spec: &mut NativeEngineSpec,
        error: &mut [c_char],
    ) -> i32;

    unsafe fn create_auto(
        &self,
        path: &CStr,
        handle: &mut *mut c_void,
        spec: &mut NativeEngineSpec,
        error: &mut [c_char],
    ) -> i32;

    unsafe fn execute(
        &self,
        handle: *mut c_void,
        input: &NativeDeviceTensorView,
        outputs: &mut [NativeHostTensorView; MAX_OUTPUTS],
        output_count: &mut u32,
        error: &mut [c_char],
    ) -> i32;

    unsafe fn probe_zero(
        &self,
        handle: *mut c_void,
        outputs: &mut [NativeHostTensorView; MAX_OUTPUTS],
        output_count: &mut u32,
        error: &mut [c_char],
    ) -> i32;

    unsafe fn destroy(&self, handle: *mut c_void);
}

#[cfg(feature = "ffi")]
#[derive(Clone, Copy, Debug)]
struct LinkedTensorRtAbi;

#[cfg(feature = "ffi")]
impl NativeTensorRtAbi for LinkedTensorRtAbi {
    fn abi_version(&self) -> u32 {
        unsafe { novasight_tensorrt_abi_version() }
    }

    unsafe fn environment(
        &self,
        environment: &mut NativeTensorRtEnvironment,
        error: &mut [c_char],
    ) -> i32 {
        unsafe {
            novasight_tensorrt_environment_query(environment, error.as_mut_ptr(), error.len())
        }
    }

    unsafe fn create(
        &self,
        path: &CStr,
        input_shape: &[u64; 4],
        handle: &mut *mut c_void,
        spec: &mut NativeEngineSpec,
        error: &mut [c_char],
    ) -> i32 {
        unsafe {
            novasight_tensorrt_create(
                path.as_ptr(),
                input_shape.as_ptr(),
                input_shape.len() as u32,
                handle,
                spec,
                error.as_mut_ptr(),
                error.len(),
            )
        }
    }

    unsafe fn create_auto(
        &self,
        path: &CStr,
        handle: &mut *mut c_void,
        spec: &mut NativeEngineSpec,
        error: &mut [c_char],
    ) -> i32 {
        unsafe {
            novasight_tensorrt_create(
                path.as_ptr(),
                std::ptr::null(),
                0,
                handle,
                spec,
                error.as_mut_ptr(),
                error.len(),
            )
        }
    }

    unsafe fn execute(
        &self,
        handle: *mut c_void,
        input: &NativeDeviceTensorView,
        outputs: &mut [NativeHostTensorView; MAX_OUTPUTS],
        output_count: &mut u32,
        error: &mut [c_char],
    ) -> i32 {
        unsafe {
            novasight_tensorrt_execute(
                handle,
                input,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
                output_count,
                error.as_mut_ptr(),
                error.len(),
            )
        }
    }

    unsafe fn probe_zero(
        &self,
        handle: *mut c_void,
        outputs: &mut [NativeHostTensorView; MAX_OUTPUTS],
        output_count: &mut u32,
        error: &mut [c_char],
    ) -> i32 {
        unsafe {
            novasight_tensorrt_probe_zero(
                handle,
                outputs.as_mut_ptr(),
                outputs.len() as u32,
                output_count,
                error.as_mut_ptr(),
                error.len(),
            )
        }
    }

    unsafe fn destroy(&self, handle: *mut c_void) {
        unsafe { novasight_tensorrt_destroy(handle) };
    }
}

#[cfg(feature = "ffi")]
unsafe extern "C" {
    fn novasight_tensorrt_abi_version() -> u32;
    fn novasight_tensorrt_environment_query(
        environment_out: *mut NativeTensorRtEnvironment,
        error_out: *mut c_char,
        error_out_size: usize,
    ) -> i32;
    fn novasight_tensorrt_create(
        engine_path: *const c_char,
        requested_input_shape: *const u64,
        requested_input_rank: u32,
        engine_out: *mut *mut c_void,
        spec_out: *mut NativeEngineSpec,
        error_out: *mut c_char,
        error_out_size: usize,
    ) -> i32;
    fn novasight_tensorrt_execute(
        engine: *mut c_void,
        input: *const NativeDeviceTensorView,
        outputs: *mut NativeHostTensorView,
        output_capacity: u32,
        output_count: *mut u32,
        error_out: *mut c_char,
        error_out_size: usize,
    ) -> i32;
    fn novasight_tensorrt_probe_zero(
        engine: *mut c_void,
        outputs: *mut NativeHostTensorView,
        output_capacity: u32,
        output_count: *mut u32,
        error_out: *mut c_char,
        error_out_size: usize,
    ) -> i32;
    fn novasight_tensorrt_destroy(engine: *mut c_void);
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    struct FakeAbi {
        output: Box<[f32; 6]>,
        executed: Mutex<Vec<NativeDeviceTensorView>>,
        destroyed: Mutex<usize>,
    }

    impl FakeAbi {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                output: Box::new([10.0, 20.0, 30.0, 40.0, 0.9, 1.0]),
                executed: Mutex::new(Vec::new()),
                destroyed: Mutex::new(0),
            })
        }
    }

    impl NativeTensorRtAbi for FakeAbi {
        fn abi_version(&self) -> u32 {
            ABI_VERSION
        }

        unsafe fn environment(
            &self,
            environment: &mut NativeTensorRtEnvironment,
            _error: &mut [c_char],
        ) -> i32 {
            environment.runtime_abi_version = ABI_VERSION;
            environment.tensorrt_runtime_version = 10_03_00;
            environment.cuda_runtime_version = 12_020;
            environment.cuda_driver_version = 12_020;
            environment.compute_capability_major = 8;
            environment.compute_capability_minor = 7;
            environment.integrated = 1;
            environment.total_global_memory = 16 * 1024 * 1024 * 1024;
            for (target, source) in environment.device_name.iter_mut().zip(b"Orin") {
                *target = *source as c_char;
            }
            0
        }

        unsafe fn create(
            &self,
            _path: &CStr,
            input_shape: &[u64; 4],
            handle: &mut *mut c_void,
            spec: &mut NativeEngineSpec,
            _error: &mut [c_char],
        ) -> i32 {
            *handle = NonNull::<u8>::dangling().as_ptr().cast();
            spec.input = native_spec("images", input_shape, 2);
            spec.output_count = 1;
            spec.outputs[0] = native_spec("output0", &[1, 1, 6], 1);
            0
        }

        unsafe fn create_auto(
            &self,
            path: &CStr,
            handle: &mut *mut c_void,
            spec: &mut NativeEngineSpec,
            error: &mut [c_char],
        ) -> i32 {
            unsafe { self.create(path, &[1, 3, 256, 256], handle, spec, error) }
        }

        unsafe fn execute(
            &self,
            _handle: *mut c_void,
            input: &NativeDeviceTensorView,
            outputs: &mut [NativeHostTensorView; MAX_OUTPUTS],
            output_count: &mut u32,
            _error: &mut [c_char],
        ) -> i32 {
            self.executed.lock().unwrap().push(*input);
            outputs[0] = NativeHostTensorView {
                host_ptr: self.output.as_ptr().cast(),
                nbytes: self.output.len() as u64 * 4,
                spec: native_spec("output0", &[1, 1, 6], 1),
            };
            *output_count = 1;
            0
        }

        unsafe fn probe_zero(
            &self,
            _handle: *mut c_void,
            outputs: &mut [NativeHostTensorView; MAX_OUTPUTS],
            output_count: &mut u32,
            _error: &mut [c_char],
        ) -> i32 {
            outputs[0] = NativeHostTensorView {
                host_ptr: self.output.as_ptr().cast(),
                nbytes: self.output.len() as u64 * 4,
                spec: native_spec("output0", &[1, 1, 6], 1),
            };
            *output_count = 1;
            0
        }

        unsafe fn destroy(&self, _handle: *mut c_void) {
            *self.destroyed.lock().unwrap() += 1;
        }
    }

    #[derive(Clone, Copy)]
    struct FakeInput {
        contract: TensorContract,
    }

    impl sealed::Sealed for FakeInput {}

    impl CudaTensorInput for FakeInput {
        fn device_ptr(&self) -> NonZeroU64 {
            NonZeroU64::new(0x1000).unwrap()
        }

        fn nbytes(&self) -> u64 {
            self.contract.nbytes()
        }

        fn contract(&self) -> TensorContract {
            self.contract
        }

        fn epoch(&self) -> RuntimeEpoch {
            RuntimeEpoch(7)
        }

        fn generation(&self) -> Generation {
            Generation(11)
        }

        fn captured_at(&self) -> MonotonicNanos {
            MonotonicNanos(123_456)
        }
    }

    fn native_spec(name: &str, dimensions: &[u64], dtype: u32) -> NativeTensorSpec {
        let mut spec = NativeTensorSpec::empty();
        for (target, source) in spec.name.iter_mut().zip(name.bytes()) {
            *target = source as c_char;
        }
        spec.rank = dimensions.len() as u32;
        for (target, source) in spec.dimensions.iter_mut().zip(dimensions.iter().copied()) {
            *target = source as i64;
        }
        spec.dtype = dtype;
        let width = if dtype == 1 { 4 } else { 2 };
        spec.nbytes = dimensions.iter().product::<u64>() * width;
        spec
    }

    #[test]
    fn environment_receipt_reports_native_runtime_and_device_identity() {
        let abi = FakeAbi::new();
        let environment = TensorRtEngine::environment_from_abi(abi.as_ref()).unwrap();
        assert_eq!(environment.runtime_abi_version(), ABI_VERSION);
        assert_eq!(environment.tensorrt_runtime_version(), 10_03_00);
        assert_eq!(environment.cuda_runtime_version(), 12_020);
        assert_eq!(environment.cuda_driver_version(), 12_020);
        assert_eq!(environment.compute_capability_major(), 8);
        assert_eq!(environment.compute_capability_minor(), 7);
        assert!(environment.integrated());
        assert_eq!(environment.device_name(), "Orin");
    }

    #[test]
    fn engine_owns_handle_and_executes_external_device_tensor_synchronously() {
        let abi = FakeAbi::new();
        let contract = TensorContract::rgb_nchw(640, 640, TensorDtype::Float16).unwrap();
        let mut engine =
            TensorRtEngine::from_abi(abi.clone(), Path::new("/models/detector.engine"), contract)
                .unwrap();
        assert_eq!(engine.contract().input().name(), "images");
        assert_eq!(engine.contract().outputs()[0].name(), "output0");

        let input = FakeInput { contract };
        {
            let outputs = engine.execute(&input).unwrap();
            assert_eq!(outputs.epoch(), RuntimeEpoch(7));
            assert_eq!(outputs.generation(), Generation(11));
            assert_eq!(outputs.captured_at(), MonotonicNanos(123_456));
            let output = outputs.iter().next().unwrap();
            assert_eq!(output.name().unwrap(), "output0");
            assert_eq!(output.dimensions(), [1, 1, 6]);
            assert_eq!(output.bytes().len(), 24);
        }
        assert_eq!(abi.executed.lock().unwrap()[0].device_ptr, 0x1000);

        drop(engine);
        assert_eq!(*abi.destroyed.lock().unwrap(), 1);
    }

    #[test]
    fn automatic_inspection_uses_resolved_shape_and_supports_a_real_probe_call() {
        let abi = FakeAbi::new();
        let mut engine =
            TensorRtEngine::inspect_from_abi(abi.clone(), Path::new("/models/detector.engine"))
                .unwrap();

        assert_eq!(engine.contract().input().dimensions(), [1, 3, 256, 256]);
        assert_eq!(engine.contract().selected_profile(), 0);
        {
            let outputs = engine.probe_zero().unwrap();
            assert_eq!(outputs.iter().next().unwrap().dimensions(), [1, 1, 6]);
        }

        drop(engine);
        assert_eq!(*abi.destroyed.lock().unwrap(), 1);
    }

    #[test]
    fn mismatched_device_tensor_is_rejected_before_native_execute() {
        let abi = FakeAbi::new();
        let loaded = TensorContract::rgb_nchw(640, 640, TensorDtype::Float16).unwrap();
        let mut engine =
            TensorRtEngine::from_abi(abi.clone(), Path::new("/models/detector.engine"), loaded)
                .unwrap();
        let wrong = FakeInput {
            contract: TensorContract::rgb_nchw(320, 320, TensorDtype::Float16).unwrap(),
        };

        assert!(matches!(
            engine.execute(&wrong),
            Err(TensorRtError::DeviceInputContractMismatch)
        ));
        assert!(abi.executed.lock().unwrap().is_empty());
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn linked_reference_library_is_versioned_and_fails_closed() {
        let contract = TensorContract::rgb_nchw(640, 640, TensorDtype::Float16).unwrap();
        let error = TensorRtEngine::load(Path::new("/models/detector.engine"), contract)
            .expect_err("portable reference must not create a runtime engine");
        assert!(matches!(
            error,
            TensorRtError::NativeCreate { code: 2, ref detail }
                if detail.contains("fail-closed")
        ));
    }

    #[test]
    fn rust_ffi_layout_matches_the_c_contract() {
        assert_eq!(std::mem::size_of::<NativeTensorSpec>(), 216);
        assert_eq!(std::mem::offset_of!(NativeTensorSpec, rank), 128);
        assert_eq!(std::mem::offset_of!(NativeTensorSpec, dimensions), 136);
        assert_eq!(std::mem::offset_of!(NativeTensorSpec, dtype), 200);
        assert_eq!(std::mem::offset_of!(NativeTensorSpec, nbytes), 208);
        assert_eq!(std::mem::size_of::<NativeEngineSpec>(), 1_960);
        assert_eq!(std::mem::offset_of!(NativeEngineSpec, outputs), 224);
        assert_eq!(std::mem::size_of::<NativeDeviceTensorView>(), 96);
        assert_eq!(std::mem::offset_of!(NativeDeviceTensorView, dimensions), 24);
        assert_eq!(std::mem::size_of::<NativeHostTensorView>(), 232);
        assert_eq!(std::mem::offset_of!(NativeHostTensorView, spec), 16);
        assert_eq!(std::mem::size_of::<NativeTensorRtEnvironment>(), 296);
        assert_eq!(
            std::mem::offset_of!(NativeTensorRtEnvironment, total_global_memory),
            32
        );
        assert_eq!(
            std::mem::offset_of!(NativeTensorRtEnvironment, device_name),
            40
        );
    }

    #[test]
    fn fp16_conversion_handles_finite_subnormal_and_special_values() {
        assert_eq!(f16_bits_to_f32(0x0000).to_bits(), 0.0_f32.to_bits());
        assert_eq!(f16_bits_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xc000), -2.0);
        assert_eq!(f16_bits_to_f32(0x0001), 2_f32.powi(-24));
        assert!(f16_bits_to_f32(0x7c00).is_infinite());
        assert!(f16_bits_to_f32(0x7e00).is_nan());
    }
}
