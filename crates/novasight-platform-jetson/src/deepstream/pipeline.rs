use std::path::{Path, PathBuf};

use novasight_core::CaptureFrameRate;
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
    pub frame_rate: CaptureFrameRate,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreviewPipelineConfig {
    pub fps: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CrosshairPipelineConfig {
    pub size: u32,
    pub fps: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InferenceStage {
    DeepStreamNvinfer { config: PathBuf },
    DeepStreamNvinferAspectPreserving { config: PathBuf },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepStreamPipelineSpec {
    pub device: PathBuf,
    pub capture: CaptureProfile,
    pub io_mode: u32,
    pub roi: Roi,
    pub model_input: ModelInput,
    pub inference: InferenceStage,
    pub batched_push_timeout_us: i64,
    pub inference_element: String,
    pub preview: Option<PreviewPipelineConfig>,
    pub crosshair: Option<CrosshairPipelineConfig>,
}

impl DeepStreamPipelineSpec {
    pub fn build(&self) -> Result<String, PipelineSpecError> {
        self.validate()?;
        let device = gst_string(&self.device);
        let mut elements = vec![format!(
            "v4l2src name=capture-source device={device} io-mode={} do-timestamp=true",
            self.io_mode
        )];
        match self.capture.format {
            CaptureFormat::Mjpeg => {
                elements.extend([
                    format!(
                        "image/jpeg,width={},height={},framerate={}/{}",
                        self.capture.width,
                        self.capture.height,
                        self.capture.frame_rate.numerator,
                        self.capture.frame_rate.denominator
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
                        "video/x-raw,format={format},width={},height={},framerate={}/{}",
                        self.capture.width,
                        self.capture.height,
                        self.capture.frame_rate.numerator,
                        self.capture.frame_rate.denominator
                    ),
                    LATEST_ONLY_QUEUE.to_owned(),
                ]);
            }
        }
        if self.preview.is_some() || self.crosshair.is_some() {
            elements.extend([
                "tee name=novasight-source-split".to_owned(),
                LATEST_ONLY_QUEUE.to_owned(),
            ]);
        }
        let right = self.roi.left + self.roi.width;
        let bottom = self.roi.top + self.roi.height;
        let preserve_roi_aspect = matches!(
            self.inference,
            InferenceStage::DeepStreamNvinferAspectPreserving { .. }
        );
        let inference_width = if preserve_roi_aspect {
            self.roi.width
        } else {
            self.model_input.width
        };
        let inference_height = if preserve_roi_aspect {
            self.roi.height
        } else {
            self.model_input.height
        };
        elements.extend([
            format!(
                "nvvidconv left={} right={right} top={} bottom={bottom}",
                self.roi.left, self.roi.top
            ),
            format!(
                "video/x-raw(memory:NVMM),format=NV12,width={},height={},pixel-aspect-ratio=1/1",
                inference_width, inference_height
            ),
            LATEST_ONLY_QUEUE.to_owned(),
        ]);
        // A request-pad reference terminates the source branch. GStreamer launch
        // syntax deliberately has no `!` between `mux.sink_0` and the named mux
        // declaration that starts the downstream branch.
        let inference = match &self.inference {
            InferenceStage::DeepStreamNvinfer { config }
            | InferenceStage::DeepStreamNvinferAspectPreserving { config } => format!(
                "nvinfer name={} config-file-path={} batch-size=1",
                self.inference_element,
                gst_string(config)
            ),
        };
        let main = format!(
            "{} ! mux.sink_0 nvstreammux name=mux batch-size=1 live-source=1 width={} height={} sync-inputs=0 batched-push-timeout={} ! {inference} ! fakesink name=deepstream-sink sync=false async=false qos=false",
            elements.join(" ! "),
            inference_width,
            inference_height,
            self.batched_push_timeout_us,
        );
        let mut branches = Vec::new();
        if let Some(preview) = self.preview {
            branches.push(format!(
                "novasight-source-split. ! {LATEST_ONLY_QUEUE} ! valve name=preview-valve drop=true ! videorate drop-only=true max-rate={} ! nvvidconv name=preview-crop left={} right={right} top={} bottom={bottom} ! video/x-raw(memory:NVMM),format=NV12,width={},height={},framerate={}/1 ! nvjpegenc name=preview-encoder ! fakesink name=preview-sink sync=false async=false qos=false",
                preview.fps, self.roi.left, self.roi.top, self.roi.width, self.roi.height, preview.fps,
            ));
        }
        if let Some(crosshair) = self.crosshair {
            let left = self.roi.left + (self.roi.width - crosshair.size) / 2;
            let top = self.roi.top + (self.roi.height - crosshair.size) / 2;
            let right = left + crosshair.size;
            let bottom = top + crosshair.size;
            branches.push(format!(
                "novasight-source-split. ! {LATEST_ONLY_QUEUE} ! videorate drop-only=true max-rate={} ! nvvidconv name=crosshair-crop left={left} right={right} top={top} bottom={bottom} ! video/x-raw(memory:NVMM),format=NV12,width={},height={},framerate={}/1 ! nvjpegenc name=crosshair-encoder ! fakesink name=crosshair-sink sync=false async=false qos=false",
                crosshair.fps, crosshair.size, crosshair.size, crosshair.fps,
            ));
        }
        Ok(std::iter::once(main)
            .chain(branches)
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn validate(&self) -> Result<(), PipelineSpecError> {
        if path_is_blank(&self.device) {
            return Err(PipelineSpecError::BlankDevice);
        }
        match &self.inference {
            InferenceStage::DeepStreamNvinfer { config }
            | InferenceStage::DeepStreamNvinferAspectPreserving { config }
                if path_is_blank(config) =>
            {
                return Err(PipelineSpecError::BlankNvinferConfig);
            }
            _ => {}
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
            (
                "capture.frame_rate.numerator",
                self.capture.frame_rate.numerator,
            ),
            (
                "capture.frame_rate.denominator",
                self.capture.frame_rate.denominator,
            ),
            ("roi.width", self.roi.width),
            ("roi.height", self.roi.height),
            ("model_input.width", self.model_input.width),
            ("model_input.height", self.model_input.height),
        ] {
            if value == 0 {
                return Err(PipelineSpecError::NonPositive { field });
            }
        }
        if self.capture.frame_rate.numerator > i32::MAX as u32
            || self.capture.frame_rate.denominator > i32::MAX as u32
        {
            return Err(PipelineSpecError::InvalidFrameRate);
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
        if self.preview.is_some_and(|preview| preview.fps == 0) {
            return Err(PipelineSpecError::ZeroPreviewFps);
        }
        if let Some(crosshair) = self.crosshair {
            if crosshair.fps == 0 {
                return Err(PipelineSpecError::ZeroCrosshairFps);
            }
            if crosshair.size < 32
                || crosshair.size > self.roi.width
                || crosshair.size > self.roi.height
            {
                return Err(PipelineSpecError::InvalidCrosshairSize {
                    size: crosshair.size,
                    roi_width: self.roi.width,
                    roi_height: self.roi.height,
                });
            }
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
    #[error("capture frame rate exceeds GStreamer fraction bounds")]
    InvalidFrameRate,
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
    #[error("preview FPS must be positive")]
    ZeroPreviewFps,
    #[error("crosshair observer FPS must be positive")]
    ZeroCrosshairFps,
    #[error(
        "crosshair observer size {size} must be at least 32 and fit ROI {roi_width}x{roi_height}"
    )]
    InvalidCrosshairSize {
        size: u32,
        roi_width: u32,
        roi_height: u32,
    },
}
