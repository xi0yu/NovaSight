use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_yaml::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub replay: ReplayConfig,
    #[serde(default)]
    pub pipeline: PipelineRuntimeConfig,
    #[serde(default)]
    pub paths: PathConfig,
    #[serde(default)]
    pub capture: Option<CaptureConfig>,
    #[serde(default)]
    pub inference: Option<InferenceConfig>,
    #[serde(default, rename = "hardware", alias = "device")]
    pub device: Option<DeviceConfig>,
    #[serde(default, flatten)]
    pub legacy: BTreeMap<String, Value>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            revision: 0,
            server: ServerConfig::default(),
            replay: ReplayConfig::default(),
            pipeline: PipelineRuntimeConfig::default(),
            paths: PathConfig::default(),
            capture: None,
            inference: None,
            device: None,
            legacy: BTreeMap::new(),
        }
    }
}

impl AppConfig {
    pub fn validate_configured_adapters(&self) -> Result<(), ConfigValidationError> {
        self.pipeline.validate()?;
        if let Some(capture) = &self.capture {
            capture.validate()?;
        }
        if let Some(inference) = &self.inference {
            inference.validate()?;
        }
        if let Some(device) = &self.device {
            device.validate()?;
        }
        Ok(())
    }

    pub fn require_production_adapters(
        &self,
    ) -> Result<ProductionAdapterConfig<'_>, ConfigValidationError> {
        self.validate_configured_adapters()?;
        let capture = self.capture.as_ref().ok_or_else(|| {
            ConfigValidationError::new("capture", "section is required in production")
        })?;
        let inference = self.inference.as_ref().ok_or_else(|| {
            ConfigValidationError::new("inference", "section is required in production")
        })?;
        let device = self.device.as_ref().ok_or_else(|| {
            ConfigValidationError::new("hardware", "section is required in production")
        })?;
        self.validate_explicit_adapter_fields()?;
        if !self.pipeline.production_fields_explicit {
            return Err(ConfigValidationError::new(
                "pipeline",
                "all production runtime fields must be explicit in YAML",
            ));
        }
        Ok(ProductionAdapterConfig {
            capture,
            inference,
            device,
            pipeline: &self.pipeline,
        })
    }

    pub(super) fn validate_explicit_adapter_fields(&self) -> Result<(), ConfigValidationError> {
        for (section, explicit) in [
            (
                "capture",
                self.capture
                    .as_ref()
                    .is_none_or(|config| config.production_fields_explicit),
            ),
            (
                "inference",
                self.inference
                    .as_ref()
                    .is_none_or(|config| config.production_fields_explicit),
            ),
            (
                "hardware",
                self.device
                    .as_ref()
                    .is_none_or(|config| config.production_fields_explicit),
            ),
        ] {
            if !explicit {
                return Err(ConfigValidationError::new(
                    section,
                    "all production adapter fields must be explicit in YAML",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProductionAdapterConfig<'a> {
    pub capture: &'a CaptureConfig,
    pub inference: &'a InferenceConfig,
    pub device: &'a DeviceConfig,
    pub pipeline: &'a PipelineRuntimeConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineRuntimeConfig {
    #[serde(default = "default_freshness_threshold_ms")]
    pub freshness_threshold_ms: f64,
    #[serde(default = "default_near_threshold_px")]
    pub near_threshold_px: f64,
    #[serde(default = "default_projection_fov_x_deg")]
    pub projection_fov_x_deg: f64,
    #[serde(default = "default_projection_counts_per_360")]
    pub projection_counts_per_360: f64,
    #[serde(default)]
    pub projection_invert_y: bool,
    #[serde(default = "default_atan_scale_counts")]
    pub atan_scale_counts: f64,
    #[serde(default = "default_far_kp")]
    pub far_kp: f64,
    #[serde(default = "default_far_max_counts_per_update")]
    pub far_max_counts_per_update: f64,
    #[serde(default = "default_near_kp")]
    pub near_kp: f64,
    #[serde(default = "default_near_max_counts_per_update")]
    pub near_max_counts_per_update: f64,
    #[serde(default = "default_velocity_smoothing_frames")]
    pub velocity_smoothing_frames: f64,
    #[serde(default = "default_velocity_history_reset_gap_ms")]
    pub velocity_history_reset_gap_ms: f64,
    #[serde(default = "default_velocity_spread_base_px_ms")]
    pub velocity_spread_base_px_ms: f64,
    #[serde(default = "default_velocity_spread_relative")]
    pub velocity_spread_relative: f64,
    #[serde(default = "default_velocity_change_base_px_ms")]
    pub velocity_change_base_px_ms: f64,
    #[serde(default = "default_velocity_change_relative")]
    pub velocity_change_relative: f64,
    #[serde(default = "default_prediction_lead_frames")]
    pub prediction_lead_frames: f64,
    #[serde(default = "default_prediction_far_absolute_cap_px")]
    pub prediction_far_absolute_cap_px: f64,
    #[serde(default = "default_prediction_far_base_cap_px")]
    pub prediction_far_base_cap_px: f64,
    #[serde(default = "default_prediction_far_relative_cap")]
    pub prediction_far_relative_cap: f64,
    #[serde(default = "default_prediction_near_absolute_cap_px")]
    pub prediction_near_absolute_cap_px: f64,
    #[serde(default = "default_prediction_near_base_cap_px")]
    pub prediction_near_base_cap_px: f64,
    #[serde(default = "default_prediction_near_relative_cap")]
    pub prediction_near_relative_cap: f64,
    #[serde(default = "default_residual_cap")]
    pub residual_cap: f64,
    #[serde(default = "default_target_debounce_distance_px")]
    pub target_debounce_distance_px: f64,
    #[serde(default = "default_target_min_confidence")]
    pub target_min_confidence: f32,
    #[serde(default = "default_target_track_max_age")]
    pub target_track_max_age: u64,
    #[serde(default = "default_tracker_max_match_distance")]
    pub tracker_max_match_distance: f64,
    #[serde(default = "default_tracker_position_cost_weight")]
    pub tracker_position_cost_weight: f64,
    #[serde(default = "default_tracker_iou_cost_weight")]
    pub tracker_iou_cost_weight: f64,
    #[serde(default = "default_max_command_age_ms")]
    pub max_command_age_ms: u64,
    #[serde(default = "default_output_interval_ms")]
    pub output_interval_ms: u64,
    #[serde(skip)]
    pub(crate) production_fields_explicit: bool,
    #[serde(default, flatten)]
    pub legacy: BTreeMap<String, Value>,
}

impl Default for PipelineRuntimeConfig {
    fn default() -> Self {
        Self {
            freshness_threshold_ms: default_freshness_threshold_ms(),
            near_threshold_px: default_near_threshold_px(),
            projection_fov_x_deg: default_projection_fov_x_deg(),
            projection_counts_per_360: default_projection_counts_per_360(),
            projection_invert_y: false,
            atan_scale_counts: default_atan_scale_counts(),
            far_kp: default_far_kp(),
            far_max_counts_per_update: default_far_max_counts_per_update(),
            near_kp: default_near_kp(),
            near_max_counts_per_update: default_near_max_counts_per_update(),
            velocity_smoothing_frames: default_velocity_smoothing_frames(),
            velocity_history_reset_gap_ms: default_velocity_history_reset_gap_ms(),
            velocity_spread_base_px_ms: default_velocity_spread_base_px_ms(),
            velocity_spread_relative: default_velocity_spread_relative(),
            velocity_change_base_px_ms: default_velocity_change_base_px_ms(),
            velocity_change_relative: default_velocity_change_relative(),
            prediction_lead_frames: default_prediction_lead_frames(),
            prediction_far_absolute_cap_px: default_prediction_far_absolute_cap_px(),
            prediction_far_base_cap_px: default_prediction_far_base_cap_px(),
            prediction_far_relative_cap: default_prediction_far_relative_cap(),
            prediction_near_absolute_cap_px: default_prediction_near_absolute_cap_px(),
            prediction_near_base_cap_px: default_prediction_near_base_cap_px(),
            prediction_near_relative_cap: default_prediction_near_relative_cap(),
            residual_cap: default_residual_cap(),
            target_debounce_distance_px: default_target_debounce_distance_px(),
            target_min_confidence: default_target_min_confidence(),
            target_track_max_age: default_target_track_max_age(),
            tracker_max_match_distance: default_tracker_max_match_distance(),
            tracker_position_cost_weight: default_tracker_position_cost_weight(),
            tracker_iou_cost_weight: default_tracker_iou_cost_weight(),
            max_command_age_ms: default_max_command_age_ms(),
            output_interval_ms: default_output_interval_ms(),
            production_fields_explicit: false,
            legacy: BTreeMap::new(),
        }
    }
}

impl PipelineRuntimeConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_finite_range(
            "pipeline.freshness_threshold_ms",
            self.freshness_threshold_ms,
            1.0,
            1_000.0,
        )?;
        validate_finite_range(
            "pipeline.near_threshold_px",
            self.near_threshold_px,
            0.0,
            10_000.0,
        )?;
        validate_finite_range(
            "pipeline.projection_fov_x_deg",
            self.projection_fov_x_deg,
            0.000_001,
            179.999_999,
        )?;
        validate_finite_range(
            "pipeline.projection_counts_per_360",
            self.projection_counts_per_360,
            0.000_001,
            1_000_000.0,
        )?;
        validate_finite_range(
            "pipeline.atan_scale_counts",
            self.atan_scale_counts,
            0.000_001,
            1_000_000.0,
        )?;
        validate_finite_range("pipeline.far_kp", self.far_kp, 0.0, 100.0)?;
        validate_finite_range(
            "pipeline.far_max_counts_per_update",
            self.far_max_counts_per_update,
            1.0,
            f64::from(i16::MAX),
        )?;
        validate_finite_range("pipeline.near_kp", self.near_kp, 0.0, 100.0)?;
        validate_finite_range(
            "pipeline.near_max_counts_per_update",
            self.near_max_counts_per_update,
            1.0,
            f64::from(i16::MAX),
        )?;
        validate_finite_range(
            "pipeline.velocity_smoothing_frames",
            self.velocity_smoothing_frames,
            0.000_001,
            1_000.0,
        )?;
        validate_finite_range(
            "pipeline.velocity_history_reset_gap_ms",
            self.velocity_history_reset_gap_ms,
            0.000_001,
            10_000.0,
        )?;
        validate_finite_range(
            "pipeline.velocity_spread_base_px_ms",
            self.velocity_spread_base_px_ms,
            0.000_001,
            10_000.0,
        )?;
        validate_finite_range(
            "pipeline.velocity_spread_relative",
            self.velocity_spread_relative,
            0.0,
            1_000.0,
        )?;
        validate_finite_range(
            "pipeline.velocity_change_base_px_ms",
            self.velocity_change_base_px_ms,
            0.000_001,
            10_000.0,
        )?;
        validate_finite_range(
            "pipeline.velocity_change_relative",
            self.velocity_change_relative,
            0.0,
            1_000.0,
        )?;
        validate_finite_range(
            "pipeline.prediction_lead_frames",
            self.prediction_lead_frames,
            0.0,
            10.0,
        )?;
        for (field, value) in [
            (
                "pipeline.prediction_far_absolute_cap_px",
                self.prediction_far_absolute_cap_px,
            ),
            (
                "pipeline.prediction_far_base_cap_px",
                self.prediction_far_base_cap_px,
            ),
            (
                "pipeline.prediction_far_relative_cap",
                self.prediction_far_relative_cap,
            ),
            (
                "pipeline.prediction_near_absolute_cap_px",
                self.prediction_near_absolute_cap_px,
            ),
            (
                "pipeline.prediction_near_base_cap_px",
                self.prediction_near_base_cap_px,
            ),
            (
                "pipeline.prediction_near_relative_cap",
                self.prediction_near_relative_cap,
            ),
        ] {
            validate_finite_range(field, value, 0.0, 100_000.0)?;
        }
        validate_finite_range("pipeline.residual_cap", self.residual_cap, 0.0, 1.0)?;
        validate_finite_range(
            "pipeline.target_debounce_distance_px",
            self.target_debounce_distance_px,
            0.000_001,
            100_000.0,
        )?;
        if !self.target_min_confidence.is_finite()
            || !(0.0..=1.0).contains(&self.target_min_confidence)
        {
            return Err(ConfigValidationError::new(
                "pipeline.target_min_confidence",
                "must be finite and within 0..=1",
            ));
        }
        if !(1..=120).contains(&self.target_track_max_age) {
            return Err(ConfigValidationError::new(
                "pipeline.target_track_max_age",
                "must be within 1..=120 frames",
            ));
        }
        validate_finite_range(
            "pipeline.tracker_max_match_distance",
            self.tracker_max_match_distance,
            0.000_001,
            100.0,
        )?;
        validate_finite_range(
            "pipeline.tracker_position_cost_weight",
            self.tracker_position_cost_weight,
            0.0,
            100.0,
        )?;
        validate_finite_range(
            "pipeline.tracker_iou_cost_weight",
            self.tracker_iou_cost_weight,
            0.0,
            100.0,
        )?;
        if self.tracker_position_cost_weight + self.tracker_iou_cost_weight <= 0.0 {
            return Err(ConfigValidationError::new(
                "pipeline.tracker_position_cost_weight",
                "tracker position and IoU weights must not both be zero",
            ));
        }
        if !(1..=1_000).contains(&self.max_command_age_ms) {
            return Err(ConfigValidationError::new(
                "pipeline.max_command_age_ms",
                "must be within 1..=1000 ms",
            ));
        }
        if !(1..=10).contains(&self.output_interval_ms) {
            return Err(ConfigValidationError::new(
                "pipeline.output_interval_ms",
                "must be within 1..=10 ms",
            ));
        }
        Ok(())
    }
}

fn validate_finite_range(
    field: &'static str,
    value: f64,
    minimum: f64,
    maximum: f64,
) -> Result<(), ConfigValidationError> {
    if !value.is_finite() || value < minimum || value > maximum {
        return Err(ConfigValidationError::new(
            field,
            format!("must be finite and within {minimum}..={maximum}"),
        ));
    }
    Ok(())
}

const fn default_freshness_threshold_ms() -> f64 {
    55.0
}

const fn default_near_threshold_px() -> f64 {
    12.0
}

const fn default_projection_fov_x_deg() -> f64 {
    105.0
}

const fn default_projection_counts_per_360() -> f64 {
    9_980.0
}

const fn default_atan_scale_counts() -> f64 {
    1_024.0
}

const fn default_far_kp() -> f64 {
    0.90
}

const fn default_far_max_counts_per_update() -> f64 {
    600.0
}

const fn default_near_kp() -> f64 {
    0.30
}

const fn default_near_max_counts_per_update() -> f64 {
    120.0
}

const fn default_velocity_smoothing_frames() -> f64 {
    3.0
}

const fn default_velocity_history_reset_gap_ms() -> f64 {
    80.0
}

const fn default_velocity_spread_base_px_ms() -> f64 {
    0.12
}

const fn default_velocity_spread_relative() -> f64 {
    0.50
}

const fn default_velocity_change_base_px_ms() -> f64 {
    0.20
}

const fn default_velocity_change_relative() -> f64 {
    0.75
}

const fn default_prediction_lead_frames() -> f64 {
    1.0
}

const fn default_prediction_far_absolute_cap_px() -> f64 {
    10.0
}

const fn default_prediction_far_base_cap_px() -> f64 {
    1.25
}

const fn default_prediction_far_relative_cap() -> f64 {
    0.30
}

const fn default_prediction_near_absolute_cap_px() -> f64 {
    3.0
}

const fn default_prediction_near_base_cap_px() -> f64 {
    0.75
}

const fn default_prediction_near_relative_cap() -> f64 {
    0.20
}

const fn default_residual_cap() -> f64 {
    1.0
}

const fn default_target_debounce_distance_px() -> f64 {
    64.0
}

const fn default_target_min_confidence() -> f32 {
    0.5
}

const fn default_target_track_max_age() -> u64 {
    5
}

const fn default_tracker_max_match_distance() -> f64 {
    1.5
}

const fn default_tracker_position_cost_weight() -> f64 {
    0.75
}

const fn default_tracker_iou_cost_weight() -> f64 {
    0.25
}

const fn default_max_command_age_ms() -> u64 {
    55
}

const fn default_output_interval_ms() -> u64 {
    4
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigValidationError {
    pub field: &'static str,
    pub message: String,
}

impl ConfigValidationError {
    fn new(field: &'static str, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.field, self.message)
    }
}

impl std::error::Error for ConfigValidationError {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaptureConfig {
    #[serde(default = "default_capture_device")]
    pub device: PathBuf,
    #[serde(default)]
    pub backend: DeepStreamBackend,
    #[serde(default)]
    pub memory: CaptureMemory,
    #[serde(default)]
    pub preference: CapturePreference,
    #[serde(default = "default_true")]
    pub latest_only: bool,
    #[serde(default = "default_one_u32")]
    pub appsink_max_buffers: u32,
    #[serde(default)]
    pub queue_leaky: QueueLeaky,
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    #[serde(default)]
    pub fps: u32,
    #[serde(default)]
    pub pixel_format: String,
    #[serde(default)]
    pub roi_left: u32,
    #[serde(default)]
    pub roi_top: u32,
    #[serde(default)]
    pub roi_width: u32,
    #[serde(default)]
    pub roi_height: u32,
    #[serde(skip)]
    pub(crate) production_fields_explicit: bool,
    #[serde(default, flatten)]
    pub legacy: BTreeMap<String, Value>,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            device: default_capture_device(),
            backend: DeepStreamBackend::default(),
            memory: CaptureMemory::default(),
            preference: CapturePreference::default(),
            latest_only: true,
            appsink_max_buffers: 1,
            queue_leaky: QueueLeaky::default(),
            width: 0,
            height: 0,
            fps: 0,
            pixel_format: String::new(),
            roi_left: 0,
            roi_top: 0,
            roi_width: 0,
            roi_height: 0,
            production_fields_explicit: false,
            legacy: BTreeMap::new(),
        }
    }
}

impl CaptureConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.device.as_os_str().is_empty() {
            return Err(ConfigValidationError::new(
                "capture.device",
                "must not be empty",
            ));
        }
        if !self.latest_only {
            return Err(ConfigValidationError::new(
                "capture.latest_only",
                "must be true",
            ));
        }
        if self.appsink_max_buffers != 1 {
            return Err(ConfigValidationError::new(
                "capture.appsink_max_buffers",
                "must be 1 for capacity-one capture",
            ));
        }
        if self.preference == CapturePreference::Manual
            && (self.pixel_format.trim().is_empty()
                || self.width == 0
                || self.height == 0
                || self.fps == 0
                || self.roi_width == 0
                || self.roi_height == 0)
        {
            return Err(ConfigValidationError::new(
                "capture.preference",
                "manual requires pixel_format, width, height, and fps",
            ));
        }
        let roi_right = self.roi_left.checked_add(self.roi_width);
        let roi_bottom = self.roi_top.checked_add(self.roi_height);
        if self.preference == CapturePreference::Manual
            && (roi_right.is_none()
                || roi_bottom.is_none()
                || roi_right.is_some_and(|right| right > self.width)
                || roi_bottom.is_some_and(|bottom| bottom > self.height))
        {
            return Err(ConfigValidationError::new(
                "capture.roi",
                "must fit inside the manual capture dimensions without overflow",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub backend: InferenceBackend,
    #[serde(default)]
    pub device: ComputeDevice,
    #[serde(default = "default_true")]
    pub require_gpu: bool,
    #[serde(default)]
    pub allow_cpu_fallback: bool,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
    #[serde(default = "default_nms_threshold")]
    pub nms_threshold: f64,
    #[serde(default = "default_inference_deadline_ms")]
    /// Zero disables this additional deadline; runtime freshness still applies.
    pub inference_input_deadline_ms: f64,
    #[serde(default = "default_parser_library")]
    pub deepstream_parser_library: PathBuf,
    #[serde(default = "default_deepstream_io_mode")]
    pub deepstream_io_mode: i32,
    #[serde(default)]
    pub deepstream_batched_push_timeout_us: i64,
    #[serde(default = "default_inference_component_id")]
    pub deepstream_component_id: i32,
    #[serde(default)]
    pub deepstream_source_id: u32,
    #[serde(default = "default_probe_element")]
    pub deepstream_probe_element: String,
    #[serde(default = "default_probe_pad")]
    pub deepstream_probe_pad: String,
    #[serde(default = "default_nvinfer_config")]
    pub deepstream_nvinfer_config: PathBuf,
    #[serde(default)]
    pub model_width: u32,
    #[serde(default)]
    pub model_height: u32,
    #[serde(default = "default_deepstream_startup_timeout_ms")]
    pub deepstream_startup_timeout_ms: u64,
    #[serde(default = "default_deepstream_shutdown_timeout_ms")]
    pub deepstream_shutdown_timeout_ms: u64,
    #[serde(default)]
    pub input_source: InferenceInputSource,
    #[serde(skip)]
    pub(crate) production_fields_explicit: bool,
    #[serde(default, flatten)]
    pub legacy: BTreeMap<String, Value>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backend: InferenceBackend::default(),
            device: ComputeDevice::default(),
            require_gpu: true,
            allow_cpu_fallback: false,
            confidence_threshold: default_confidence_threshold(),
            nms_threshold: default_nms_threshold(),
            inference_input_deadline_ms: default_inference_deadline_ms(),
            deepstream_parser_library: default_parser_library(),
            deepstream_io_mode: default_deepstream_io_mode(),
            deepstream_batched_push_timeout_us: 0,
            deepstream_component_id: default_inference_component_id(),
            deepstream_source_id: 0,
            deepstream_probe_element: default_probe_element(),
            deepstream_probe_pad: default_probe_pad(),
            deepstream_nvinfer_config: default_nvinfer_config(),
            model_width: 0,
            model_height: 0,
            deepstream_startup_timeout_ms: default_deepstream_startup_timeout_ms(),
            deepstream_shutdown_timeout_ms: default_deepstream_shutdown_timeout_ms(),
            input_source: InferenceInputSource::default(),
            production_fields_explicit: false,
            legacy: BTreeMap::new(),
        }
    }
}

impl InferenceConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !self.require_gpu || self.allow_cpu_fallback {
            return Err(ConfigValidationError::new(
                "inference.device",
                "must require CUDA with CPU fallback disabled",
            ));
        }
        for (field, value) in [
            ("inference.confidence_threshold", self.confidence_threshold),
            ("inference.nms_threshold", self.nms_threshold),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(ConfigValidationError::new(
                    field,
                    "must be finite and in [0, 1]",
                ));
            }
        }
        if !self.inference_input_deadline_ms.is_finite() || self.inference_input_deadline_ms <= 0.0
        {
            return Err(ConfigValidationError::new(
                "inference.inference_input_deadline_ms",
                "must be finite and positive so stale batches cannot unlock runtime readiness",
            ));
        }
        if self.backend == InferenceBackend::DeepstreamNvinfer
            && self
                .deepstream_parser_library
                .to_string_lossy()
                .trim()
                .is_empty()
        {
            return Err(ConfigValidationError::new(
                "inference.deepstream_parser_library",
                "must not be empty",
            ));
        }
        if self.deepstream_component_id < 0 {
            return Err(ConfigValidationError::new(
                "inference.deepstream_component_id",
                "must be non-negative",
            ));
        }
        if self.deepstream_io_mode < 0 || self.deepstream_batched_push_timeout_us < 0 {
            return Err(ConfigValidationError::new(
                "inference.deepstream_io_mode",
                "I/O mode and batched push timeout must be non-negative",
            ));
        }
        if self.deepstream_probe_element.trim().is_empty()
            || self.deepstream_probe_pad.trim().is_empty()
        {
            return Err(ConfigValidationError::new(
                "inference.deepstream_probe_element",
                "probe element and pad must not be empty",
            ));
        }
        if self.production_fields_explicit
            && self.backend == InferenceBackend::DeepstreamNvinfer
            && self
                .deepstream_nvinfer_config
                .to_string_lossy()
                .trim()
                .is_empty()
        {
            return Err(ConfigValidationError::new(
                "inference.deepstream_nvinfer_config",
                "must not be empty",
            ));
        }
        if self.production_fields_explicit && (self.model_width == 0 || self.model_height == 0) {
            return Err(ConfigValidationError::new(
                "inference.model_width",
                "model width and height must be positive",
            ));
        }
        if self.production_fields_explicit
            && (self.deepstream_startup_timeout_ms == 0 || self.deepstream_shutdown_timeout_ms == 0)
        {
            return Err(ConfigValidationError::new(
                "inference.deepstream_startup_timeout_ms",
                "startup and shutdown timeouts must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepStreamBackend {
    #[default]
    DeepstreamNvinfer,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackend {
    #[default]
    DeepstreamNvinfer,
    RustTensorRt,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaptureMemory {
    #[default]
    Nvmm,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapturePreference {
    #[default]
    AutoHighFps,
    AutoLowLatency,
    AutoBalanced,
    Manual,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QueueLeaky {
    #[default]
    Downstream,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeDevice {
    #[default]
    Cuda,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum InferenceInputSource {
    #[default]
    #[serde(rename = "source.default")]
    SourceDefault,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceConfig {
    #[serde(default)]
    pub auto_connect: bool,
    #[serde(default)]
    pub backend: DeviceBackend,
    #[serde(default = "default_kmnet_host")]
    pub host: String,
    #[serde(default = "default_kmnet_port")]
    pub port: u16,
    #[serde(default = "default_kmnet_uuid")]
    pub uuid: String,
    #[serde(default = "default_kmnet_monitor_port")]
    pub monitor_port: u16,
    #[serde(default = "default_kmnet_helper_module")]
    pub helper_module: String,
    #[serde(default = "default_kmnet_connect_timeout_ms")]
    pub connect_timeout_ms: u64,
    #[serde(default = "default_kmnet_send_timeout_ms")]
    pub send_timeout_ms: u64,
    #[serde(default = "default_kmnet_monitor_timeout_ms")]
    pub monitor_timeout_ms: u64,
    #[serde(default = "default_kmnet_trigger_poll_interval_ms")]
    pub trigger_poll_interval_ms: u64,
    #[serde(default = "default_kmnet_reconnect_cooldown_ms")]
    pub reconnect_cooldown_ms: u64,
    #[serde(skip)]
    pub(crate) production_fields_explicit: bool,
    #[serde(default, flatten)]
    pub legacy: BTreeMap<String, Value>,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            auto_connect: false,
            backend: DeviceBackend::default(),
            host: default_kmnet_host(),
            port: default_kmnet_port(),
            uuid: default_kmnet_uuid(),
            monitor_port: default_kmnet_monitor_port(),
            helper_module: default_kmnet_helper_module(),
            connect_timeout_ms: default_kmnet_connect_timeout_ms(),
            send_timeout_ms: default_kmnet_send_timeout_ms(),
            monitor_timeout_ms: default_kmnet_monitor_timeout_ms(),
            trigger_poll_interval_ms: default_kmnet_trigger_poll_interval_ms(),
            reconnect_cooldown_ms: default_kmnet_reconnect_cooldown_ms(),
            production_fields_explicit: false,
            legacy: BTreeMap::new(),
        }
    }
}

impl DeviceConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.auto_connect && self.host.trim().is_empty() {
            return Err(ConfigValidationError::new(
                "hardware.host",
                "must not be empty",
            ));
        }
        if self.auto_connect && self.uuid.trim().is_empty() {
            return Err(ConfigValidationError::new(
                "hardware.uuid",
                "must not be empty",
            ));
        }
        if self.auto_connect && self.port == 0 {
            return Err(ConfigValidationError::new(
                "hardware.port",
                "must be non-zero when auto_connect is enabled",
            ));
        }
        if self.auto_connect && !(1024..=49_151).contains(&self.monitor_port) {
            return Err(ConfigValidationError::new(
                "hardware.monitor_port",
                "must be within the kmNet vendor range 1024..=49151",
            ));
        }
        if self.auto_connect
            && self.backend == DeviceBackend::PythonHost
            && self.helper_module.trim().is_empty()
        {
            return Err(ConfigValidationError::new(
                "hardware.helper_module",
                "must not be empty",
            ));
        }
        if self.auto_connect && self.connect_timeout_ms == 0 {
            return Err(ConfigValidationError::new(
                "hardware.connect_timeout_ms",
                "must be non-zero",
            ));
        }
        if self.auto_connect && self.send_timeout_ms == 0 {
            return Err(ConfigValidationError::new(
                "hardware.send_timeout_ms",
                "must be non-zero",
            ));
        }
        if self.auto_connect && !(1..=50).contains(&self.trigger_poll_interval_ms) {
            return Err(ConfigValidationError::new(
                "hardware.trigger_poll_interval_ms",
                "must be within 1..=50",
            ));
        }
        if self.auto_connect
            && self.backend == DeviceBackend::NativeUdp
            && self.monitor_timeout_ms == 0
        {
            return Err(ConfigValidationError::new(
                "hardware.monitor_timeout_ms",
                "must be non-zero for native_udp",
            ));
        }
        if self.auto_connect
            && self.backend == DeviceBackend::PythonHost
            && self.reconnect_cooldown_ms == 0
        {
            return Err(ConfigValidationError::new(
                "hardware.reconnect_cooldown_ms",
                "must be non-zero for python_host",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceBackend {
    NativeUdp,
    #[default]
    PythonHost,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_server_host")]
    pub host: String,
    #[serde(default = "default_server_port")]
    pub port: u16,
    #[serde(default = "default_control_socket")]
    pub control_socket: PathBuf,
    #[serde(default, flatten)]
    pub legacy: BTreeMap<String, Value>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_server_host(),
            port: default_server_port(),
            control_socket: default_control_socket(),
            legacy: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayConfig {
    #[serde(default = "default_replay_enabled")]
    pub enabled: bool,
    #[serde(default = "default_frame_interval_ms")]
    pub frame_interval_ms: u64,
    #[serde(default)]
    pub output_gate_open: bool,
    #[serde(default, flatten)]
    pub legacy: BTreeMap<String, Value>,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            enabled: default_replay_enabled(),
            frame_interval_ms: default_frame_interval_ms(),
            output_gate_open: false,
            legacy: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    #[serde(default = "default_model_dir")]
    pub model_dir: PathBuf,
    #[serde(default = "default_database")]
    pub database: PathBuf,
    #[serde(default = "default_license")]
    pub license: PathBuf,
    #[serde(default = "default_python_executable")]
    pub python_executable: PathBuf,
    #[serde(default, flatten)]
    pub legacy: BTreeMap<String, Value>,
}

impl Default for PathConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            model_dir: default_model_dir(),
            database: default_database(),
            license: default_license(),
            python_executable: default_python_executable(),
            legacy: BTreeMap::new(),
        }
    }
}

const fn default_schema_version() -> u32 {
    1
}

fn default_server_host() -> String {
    "0.0.0.0".to_owned()
}

const fn default_server_port() -> u16 {
    5174
}

fn default_control_socket() -> PathBuf {
    PathBuf::from("/run/novasight/novasightd.sock")
}

const fn default_replay_enabled() -> bool {
    true
}

const fn default_frame_interval_ms() -> u64 {
    16
}

fn default_data_dir() -> PathBuf {
    PathBuf::from("data")
}

fn default_model_dir() -> PathBuf {
    PathBuf::from("data/models")
}

fn default_database() -> PathBuf {
    PathBuf::from("data/novasight.db")
}

fn default_license() -> PathBuf {
    PathBuf::from("data/license.json")
}

fn default_python_executable() -> PathBuf {
    PathBuf::from("python3")
}

fn default_capture_device() -> PathBuf {
    PathBuf::from("/dev/video0")
}

const fn default_true() -> bool {
    true
}

const fn default_one_u32() -> u32 {
    1
}

const fn default_confidence_threshold() -> f64 {
    0.25
}

const fn default_nms_threshold() -> f64 {
    0.45
}

const fn default_inference_deadline_ms() -> f64 {
    55.0
}

fn default_parser_library() -> PathBuf {
    PathBuf::from("build/deepstream-parser/libnovasight_parser.so")
}

const fn default_deepstream_io_mode() -> i32 {
    2
}

const fn default_inference_component_id() -> i32 {
    1
}

fn default_probe_element() -> String {
    "primary-infer".to_owned()
}

fn default_probe_pad() -> String {
    "src".to_owned()
}

fn default_nvinfer_config() -> PathBuf {
    PathBuf::from("data/runtime/deepstream/active-nvinfer.ini")
}

const fn default_deepstream_startup_timeout_ms() -> u64 {
    10_000
}

const fn default_deepstream_shutdown_timeout_ms() -> u64 {
    5_000
}

fn default_kmnet_host() -> String {
    "192.168.2.188".to_owned()
}

const fn default_kmnet_port() -> u16 {
    8888
}

fn default_kmnet_uuid() -> String {
    "12345678".to_owned()
}

const fn default_kmnet_monitor_port() -> u16 {
    5001
}

fn default_kmnet_helper_module() -> String {
    "novasight.executors.kmnet_host".to_owned()
}

const fn default_kmnet_connect_timeout_ms() -> u64 {
    3_000
}

const fn default_kmnet_send_timeout_ms() -> u64 {
    25
}

const fn default_kmnet_monitor_timeout_ms() -> u64 {
    250
}

const fn default_kmnet_trigger_poll_interval_ms() -> u64 {
    4
}

const fn default_kmnet_reconnect_cooldown_ms() -> u64 {
    500
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_backend_has_an_explicit_rust_tensorrt_value() {
        assert_eq!(
            serde_json::from_str::<InferenceBackend>(r#""rust_tensor_rt""#).unwrap(),
            InferenceBackend::RustTensorRt
        );
        assert_eq!(
            serde_json::to_string(&InferenceBackend::DeepstreamNvinfer).unwrap(),
            r#""deepstream_nvinfer""#
        );
    }

    #[test]
    fn rust_tensorrt_backend_does_not_require_nvinfer_artifacts() {
        let config = InferenceConfig {
            backend: InferenceBackend::RustTensorRt,
            deepstream_parser_library: PathBuf::new(),
            deepstream_nvinfer_config: PathBuf::new(),
            model_width: 640,
            model_height: 640,
            production_fields_explicit: true,
            ..InferenceConfig::default()
        };
        config.validate().unwrap();
    }

    #[test]
    fn tracker_association_requires_at_least_one_real_cost_signal() {
        let config = PipelineRuntimeConfig {
            tracker_position_cost_weight: 0.0,
            tracker_iou_cost_weight: 0.0,
            ..PipelineRuntimeConfig::default()
        };
        let error = config.validate().expect_err("zero association weights");
        assert_eq!(error.field, "pipeline.tracker_position_cost_weight");
    }
}
