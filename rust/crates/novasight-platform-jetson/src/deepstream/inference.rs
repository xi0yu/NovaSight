use std::path::Path;

use novasight_tensorrt::{ExecutionOutputs, TensorRtEngine, TensorRtError};
use thiserror::Error;

use super::{CudaFramePreprocessor, FrameLease, PreprocessError, TensorContract};

/// Thread-affine owner for one complete synchronous GPU inference interval.
///
/// Construct this value inside the dedicated inference thread. Its TensorRT
/// execution context, CUDA stream, pinned outputs, and transient DeviceTensor
/// never cross that owner thread.
pub struct CudaTensorRtOwner {
    preprocess: CudaFramePreprocessor,
    engine: TensorRtEngine,
    input: TensorContract,
}

impl std::fmt::Debug for CudaTensorRtOwner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaTensorRtOwner")
            .field("preprocess_backend", &self.preprocess.backend())
            .field("engine", &self.engine)
            .field("input", &self.input)
            .finish()
    }
}

impl CudaTensorRtOwner {
    pub fn load(engine_path: &Path, input: TensorContract) -> Result<Self, CudaTensorRtOwnerError> {
        let preprocess = CudaFramePreprocessor::linked()?;
        let engine = TensorRtEngine::load(engine_path, input)?;
        Ok(Self {
            preprocess,
            engine,
            input,
        })
    }

    pub fn engine_contract(&self) -> &novasight_tensorrt::EngineContract {
        self.engine.contract()
    }

    pub fn execute<'owner>(
        &'owner mut self,
        frame: &FrameLease,
    ) -> Result<ExecutionOutputs<'owner>, CudaTensorRtOwnerError> {
        let tensor = self.preprocess.prepare(frame, self.input)?;
        // Native execute synchronizes the owner stream before returning. The
        // input allocation can therefore be released immediately while the
        // returned views borrow TensorRT-owned pinned host outputs.
        Ok(self.engine.execute(&tensor)?)
    }
}

#[derive(Debug, Error)]
pub enum CudaTensorRtOwnerError {
    #[error(transparent)]
    Preprocess(#[from] PreprocessError),
    #[error(transparent)]
    TensorRt(#[from] TensorRtError),
}
