use novasight_jetson_preprocess::{
    CudaPreprocessor, DeviceTensor, NvmmFrameInput, PreprocessError, TensorContract,
};

use super::FrameLease;

/// Deep adapter from a strongly-held GStreamer/NVMM frame to a model-ready
/// CUDA tensor. The native JSON ABI and allocation token never cross this seam.
#[derive(Clone, Debug)]
pub struct CudaFramePreprocessor {
    inner: CudaPreprocessor,
}

impl CudaFramePreprocessor {
    pub fn linked() -> Result<Self, PreprocessError> {
        Ok(Self {
            inner: CudaPreprocessor::linked()?,
        })
    }

    pub fn backend(&self) -> &str {
        self.inner.backend()
    }

    pub fn prepare(
        &self,
        frame: &FrameLease,
        contract: TensorContract,
    ) -> Result<DeviceTensor, PreprocessError> {
        let (width, height) = frame.dimensions();
        let input = NvmmFrameInput::from_deepstream_pad(
            frame.epoch(),
            frame.generation(),
            frame.captured_at(),
            frame.buffer_ptr(),
            width,
            height,
        )?;
        // The native implementation synchronizes its CUDA work before
        // returning, so this borrow keeps GstBuffer/NVMM alive for the complete
        // map/transform/kernel interval. DeviceTensor owns only NovaSight CUDA
        // memory after this call.
        self.inner.prepare(input, contract)
    }
}
