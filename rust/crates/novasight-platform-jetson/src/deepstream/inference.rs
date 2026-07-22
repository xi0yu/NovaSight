use std::path::{Path, PathBuf};

use novasight_core::DetectionBatch;
use novasight_tensorrt::{DecodeError, DetectionDecoder, TensorRtEngine, TensorRtError};
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
    decoder: DetectionDecoder,
    input: TensorContract,
}

#[derive(Clone, Debug)]
pub struct CudaTensorRtConfig {
    pub engine_path: PathBuf,
    pub input: TensorContract,
    pub decoder: DetectionDecoder,
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
    pub fn from_config(config: CudaTensorRtConfig) -> Result<Self, CudaTensorRtOwnerError> {
        Self::load(&config.engine_path, config.input, config.decoder)
    }

    pub fn load(
        engine_path: &Path,
        input: TensorContract,
        decoder: DetectionDecoder,
    ) -> Result<Self, CudaTensorRtOwnerError> {
        let preprocess = CudaFramePreprocessor::linked()?;
        let engine = TensorRtEngine::load(engine_path, input)?;
        decoder.validate_engine_contract(engine.contract())?;
        Ok(Self {
            preprocess,
            engine,
            decoder,
            input,
        })
    }

    pub fn engine_contract(&self) -> &novasight_tensorrt::EngineContract {
        self.engine.contract()
    }

    pub fn infer(&mut self, frame: &FrameLease) -> Result<DetectionBatch, CudaTensorRtOwnerError> {
        let tensor = self.preprocess.prepare(frame, self.input)?;
        // Native execute synchronizes the owner stream before returning. The
        // input allocation can therefore be released immediately while the
        // returned views borrow TensorRT-owned pinned host outputs.
        let outputs = self.engine.execute(&tensor)?;
        Ok(self.decoder.decode(&outputs)?)
    }
}

#[derive(Debug, Error)]
pub enum CudaTensorRtOwnerError {
    #[error(transparent)]
    Preprocess(#[from] PreprocessError),
    #[error(transparent)]
    TensorRt(#[from] TensorRtError),
    #[error(transparent)]
    Decode(#[from] DecodeError),
}
