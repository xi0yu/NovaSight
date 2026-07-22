use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CaptureCapability {
    pub pixel_format: String,
    pub width: u32,
    pub height: u32,
    pub fps_list: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CaptureCapabilities {
    pub available: bool,
    pub device: String,
    pub capabilities: Vec<CaptureCapability>,
    pub reason: String,
}

pub trait CaptureCapabilityProbe: Send + Sync + 'static {
    fn probe(&self, device: &str) -> Result<CaptureCapabilities, CaptureProbeError>;
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{message}")]
pub struct CaptureProbeError {
    code: &'static str,
    message: String,
}

impl CaptureProbeError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub const fn code(&self) -> &'static str {
        self.code
    }
}
