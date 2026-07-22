use std::path::{Path, PathBuf};

use thiserror::Error;

const LATEST_ONLY_QUEUE: &str =
    "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureFormat {
    Mjpeg,
    Nv12,
    Yuy2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CaptureProfile {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub format: CaptureFormat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Roi {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelInput {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepStreamPipelineSpec {
    pub device: PathBuf,
    pub capture: CaptureProfile,
    pub io_mode: u32,
    pub roi: Roi,
    pub model_input: ModelInput,
    pub nvinfer_config: PathBuf,
    pub batched_push_timeout_us: i64,
    pub inference_element: String,
}

impl DeepStreamPipelineSpec {
    pub fn build(&self) -> Result<String, PipelineSpecError> {
        self.validate()?;
        let device = gst_string(&self.device);
        let nvinfer_config = gst_string(&self.nvinfer_config);
        let mut elements = vec![format!(
            "v4l2src name=capture-source device={device} io-mode={} do-timestamp=true",
            self.io_mode
        )];
        match self.capture.format {
            CaptureFormat::Mjpeg => {
                elements.extend([
                    format!(
                        "image/jpeg,width={},height={},framerate={}/1",
                        self.capture.width, self.capture.height, self.capture.fps
                    ),
                    LATEST_ONLY_QUEUE.to_owned(),
                    "jpegparse".to_owned(),
                    "nvv4l2decoder mjpeg=1".to_owned(),
                    "video/x-raw(memory:NVMM),format=I420".to_owned(),
                    LATEST_ONLY_QUEUE.to_owned(),
                ]);
            }
            CaptureFormat::Nv12 | CaptureFormat::Yuy2 => {
                let format = match self.capture.format {
                    CaptureFormat::Nv12 => "NV12",
                    CaptureFormat::Yuy2 => "YUY2",
                    CaptureFormat::Mjpeg => unreachable!(),
                };
                elements.extend([
                    format!(
                        "video/x-raw,format={format},width={},height={},framerate={}/1",
                        self.capture.width, self.capture.height, self.capture.fps
                    ),
                    LATEST_ONLY_QUEUE.to_owned(),
                ]);
            }
        }
        let right = self.roi.left + self.roi.width;
        let bottom = self.roi.top + self.roi.height;
        elements.extend([
            format!(
                "nvvidconv left={} right={right} top={} bottom={bottom}",
                self.roi.left, self.roi.top
            ),
            format!(
                "video/x-raw(memory:NVMM),format=NV12,width={},height={},pixel-aspect-ratio=1/1",
                self.model_input.width, self.model_input.height
            ),
            LATEST_ONLY_QUEUE.to_owned(),
        ]);
        // A request-pad reference terminates the source branch. GStreamer launch
        // syntax deliberately has no `!` between `mux.sink_0` and the named mux
        // declaration that starts the downstream branch.
        Ok(format!(
            "{} ! mux.sink_0 nvstreammux name=mux batch-size=1 live-source=1 width={} height={} sync-inputs=0 batched-push-timeout={} ! nvinfer name={} config-file-path={nvinfer_config} batch-size=1 ! fakesink name=deepstream-sink sync=false async=false qos=false",
            elements.join(" ! "),
            self.model_input.width,
            self.model_input.height,
            self.batched_push_timeout_us,
            self.inference_element,
        ))
    }

    fn validate(&self) -> Result<(), PipelineSpecError> {
        if path_is_blank(&self.device) {
            return Err(PipelineSpecError::BlankDevice);
        }
        if path_is_blank(&self.nvinfer_config) {
            return Err(PipelineSpecError::BlankNvinferConfig);
        }
        if self.inference_element.trim().is_empty() {
            return Err(PipelineSpecError::BlankInferenceElement);
        }
        if !self
            .inference_element
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(PipelineSpecError::InvalidInferenceElement);
        }
        if self.io_mode > 5 {
            return Err(PipelineSpecError::InvalidIoMode(self.io_mode));
        }
        for (field, value) in [
            ("capture.width", self.capture.width),
            ("capture.height", self.capture.height),
            ("capture.fps", self.capture.fps),
            ("roi.width", self.roi.width),
            ("roi.height", self.roi.height),
            ("model_input.width", self.model_input.width),
            ("model_input.height", self.model_input.height),
        ] {
            if value == 0 {
                return Err(PipelineSpecError::NonPositive { field });
            }
        }
        let right = self
            .roi
            .left
            .checked_add(self.roi.width)
            .ok_or(PipelineSpecError::RoiOverflow)?;
        let bottom = self
            .roi
            .top
            .checked_add(self.roi.height)
            .ok_or(PipelineSpecError::RoiOverflow)?;
        if right > self.capture.width || bottom > self.capture.height {
            return Err(PipelineSpecError::RoiOutsideCapture {
                capture_width: self.capture.width,
                capture_height: self.capture.height,
                right,
                bottom,
            });
        }
        if self.batched_push_timeout_us < 0 {
            return Err(PipelineSpecError::NegativeBatchedPushTimeout(
                self.batched_push_timeout_us,
            ));
        }
        Ok(())
    }
}

fn path_is_blank(path: &Path) -> bool {
    path.to_string_lossy().trim().is_empty()
}

fn gst_string(path: &Path) -> String {
    let value = path.to_string_lossy();
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PipelineSpecError {
    #[error("capture device must not be blank")]
    BlankDevice,
    #[error("nvinfer config path must not be blank")]
    BlankNvinferConfig,
    #[error("inference element name must not be blank")]
    BlankInferenceElement,
    #[error("inference element name may contain only ASCII letters, digits, '-' and '_'")]
    InvalidInferenceElement,
    #[error("V4L2 io-mode must be in 0..=5, got {0}")]
    InvalidIoMode(u32),
    #[error("{field} must be positive")]
    NonPositive { field: &'static str },
    #[error("ROI coordinate overflow")]
    RoiOverflow,
    #[error("ROI right={right} bottom={bottom} exceeds capture {capture_width}x{capture_height}")]
    RoiOutsideCapture {
        capture_width: u32,
        capture_height: u32,
        right: u32,
        bottom: u32,
    },
    #[error("batched push timeout must be non-negative, got {0}")]
    NegativeBatchedPushTimeout(i64),
}
