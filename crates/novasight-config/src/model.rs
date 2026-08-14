use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use novasight_core::tracking::KalmanConfig;
use serde::{Deserialize, Serialize};
use serde_yaml::Value;

pub const CURRENT_SCHEMA_VERSION: u32 = 15;

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
    pub control: RustControlConfig,
    #[serde(default)]
    pub paths: PathConfig,
    #[serde(default)]
    pub consumers: ConsumerConfig,
    #[serde(default)]
    pub crosshair: CrosshairConfig,
    #[serde(default)]
    pub limits: LimitsConfig,
    #[serde(default)]
    pub capture: Option<CaptureConfig>,
    #[serde(default)]
    pub inference: Option<InferenceConfig>,
    #[serde(default, rename = "hardware", alias = "device")]
    pub device: Option<DeviceConfig>,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            revision: 0,
            server: ServerConfig::default(),
            replay: ReplayConfig::default(),
            pipeline: PipelineRuntimeConfig::default(),
            control: RustControlConfig::default(),
            paths: PathConfig::default(),
            consumers: ConsumerConfig::default(),
            crosshair: CrosshairConfig::default(),
            limits: LimitsConfig::default(),
            capture: None,
            inference: None,
            device: None,
            extra: BTreeMap::new(),
        }
    }
}

impl AppConfig {
    /// Validate only the model-inference seam used by candidate publication.
    /// Capture, control, server, and pointer-device policy belong to runtime
    /// composition and must not make an otherwise valid Engine unpublishable.
    pub fn require_inference_adapter(&self) -> Result<&InferenceConfig, ConfigValidationError> {
        let inference = self.inference.as_ref().ok_or_else(|| {
            ConfigValidationError::new("inference", "section is required for model inference")
        })?;
        inference.validate()?;
        Ok(inference)
    }

    pub fn validate_configured_adapters(&self) -> Result<(), ConfigValidationError> {
        self.pipeline.validate()?;
        self.control.recoil.validate()?;
        if let Some(capture) = &self.capture {
            capture.validate()?;
        }
        if let Some(inference) = &self.inference {
            inference.validate()?;
        }
        if let Some(device) = &self.device {
            device.validate()?;
            if self.control.output_enabled && !device.auto_connect {
                return Err(ConfigValidationError::new(
                    "control.output_enabled",
                    "cannot be true until hardware.auto_connect is enabled with a commissioned device",
                ));
            }
        } else if self.control.output_enabled {
            return Err(ConfigValidationError::new(
                "control.output_enabled",
                "cannot be true until a commissioned hardware section is configured",
            ));
        }
        self.crosshair.validate()?;
        if self.limits.stream_fps == 0 {
            return Err(ConfigValidationError::new(
                "limits.stream_fps",
                "must be positive",
            ));
        }
        Ok(())
    }

    pub fn require_production_adapters(
        &self,
    ) -> Result<ProductionAdapterConfig<'_>, ConfigValidationError> {
        let vision = self.require_vision_adapters()?;
        let device = self.device.as_ref().ok_or_else(|| {
            ConfigValidationError::new(
                "hardware",
                "section is required when composing a commissioned output adapter",
            )
        })?;
        Ok(ProductionAdapterConfig {
            capture: vision.capture,
            inference: vision.inference,
            device,
            pipeline: vision.pipeline,
            consumers: vision.consumers,
            limits: vision.limits,
        })
    }

    /// Compose the headless capture/inference/control runtime without forcing
    /// callers to invent a physical pointer-device configuration.
    pub fn require_vision_adapters(
        &self,
    ) -> Result<VisionAdapterConfig<'_>, ConfigValidationError> {
        self.pipeline.validate()?;
        self.control.recoil.validate()?;
        self.crosshair.validate()?;
        if self.limits.stream_fps == 0 {
            return Err(ConfigValidationError::new(
                "limits.stream_fps",
                "must be positive",
            ));
        }
        let capture = self.capture.as_ref().ok_or_else(|| {
            ConfigValidationError::new("capture", "section is required for live vision")
        })?;
        let inference = self.inference.as_ref().ok_or_else(|| {
            ConfigValidationError::new("inference", "section is required for live vision")
        })?;
        capture.validate()?;
        inference.validate()?;
        if !capture.production_fields_explicit {
            return Err(ConfigValidationError::new(
                "capture",
                "all capture intent fields must be explicit in YAML",
            ));
        }
        if !inference.production_fields_explicit {
            return Err(ConfigValidationError::new(
                "inference",
                "all inference intent fields must be explicit in YAML",
            ));
        }
        if !self.pipeline.production_fields_explicit {
            return Err(ConfigValidationError::new(
                "pipeline",
                "all production runtime fields must be explicit in YAML",
            ));
        }
        Ok(VisionAdapterConfig {
            capture,
            inference,
            pipeline: &self.pipeline,
            consumers: &self.consumers,
            limits: &self.limits,
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RustControlConfig {
    #[serde(default)]
    pub output_enabled: bool,
    #[serde(default)]
    pub trigger_mode: TriggerMode,
    #[serde(default)]
    pub recoil: RecoilConfig,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Determines whether a valid target is sufficient to activate movement or
/// whether the commissioned device must also report a held hardware trigger.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerMode {
    #[default]
    Always,
    Hardware,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecoilConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_recoil_require_target")]
    pub require_target: bool,
    #[serde(default = "default_recoil_interval_ms")]
    pub interval_ms: u64,
    #[serde(default)]
    pub fire_delay_enabled: bool,
    #[serde(default = "default_recoil_fire_delay_ms")]
    pub fire_delay_ms: u64,
    #[serde(default = "default_recoil_y_counts")]
    pub y_counts: i32,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Default for RecoilConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            require_target: default_recoil_require_target(),
            interval_ms: default_recoil_interval_ms(),
            fire_delay_enabled: false,
            fire_delay_ms: default_recoil_fire_delay_ms(),
            y_counts: default_recoil_y_counts(),
            extra: BTreeMap::new(),
        }
    }
}

impl RecoilConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !(1..=5_000).contains(&self.interval_ms) {
            return Err(ConfigValidationError::new(
                "control.recoil.interval_ms",
                "must be within 1..=5000 ms",
            ));
        }
        if self.fire_delay_ms > 5_000 {
            return Err(ConfigValidationError::new(
                "control.recoil.fire_delay_ms",
                "must be within 0..=5000 ms",
            ));
        }
        if !(1..=i16::MAX as i32).contains(&self.y_counts) {
            return Err(ConfigValidationError::new(
                "control.recoil.y_counts",
                "must be within 1..=32767 positive-Y counts",
            ));
        }
        Ok(())
    }
}

const fn default_recoil_require_target() -> bool {
    true
}
const fn default_recoil_interval_ms() -> u64 {
    16
}
const fn default_recoil_fire_delay_ms() -> u64 {
    0
}
const fn default_recoil_y_counts() -> i32 {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrosshairConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub use_for_control: bool,
    #[serde(default = "default_crosshair_search_size")]
    pub search_size: u32,
    #[serde(default = "default_crosshair_sample_hz")]
    pub sample_hz: u32,
    #[serde(default = "default_crosshair_sample_frames")]
    pub sample_frames: usize,
    #[serde(default = "default_crosshair_confirm_duration_ms")]
    pub confirm_duration_ms: f64,
    #[serde(default = "default_crosshair_max_age_ms")]
    pub max_age_ms: f64,
    #[serde(default = "default_crosshair_max_offset_px")]
    pub max_offset_px: f64,
    #[serde(default = "default_crosshair_min_similarity")]
    pub min_similarity: f64,
    #[serde(default = "default_crosshair_max_step_px")]
    pub max_step_px: f64,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Default for CrosshairConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            use_for_control: false,
            search_size: default_crosshair_search_size(),
            sample_hz: default_crosshair_sample_hz(),
            sample_frames: default_crosshair_sample_frames(),
            confirm_duration_ms: default_crosshair_confirm_duration_ms(),
            max_age_ms: default_crosshair_max_age_ms(),
            max_offset_px: default_crosshair_max_offset_px(),
            min_similarity: default_crosshair_min_similarity(),
            max_step_px: default_crosshair_max_step_px(),
            extra: BTreeMap::new(),
        }
    }
}

impl CrosshairConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if !(32..=640).contains(&self.search_size) {
            return Err(ConfigValidationError::new(
                "crosshair.search_size",
                "must be within 32..=640",
            ));
        }
        if !(1..=60).contains(&self.sample_hz) {
            return Err(ConfigValidationError::new(
                "crosshair.sample_hz",
                "must be within 1..=60",
            ));
        }
        if !(3..=31).contains(&self.sample_frames) {
            return Err(ConfigValidationError::new(
                "crosshair.sample_frames",
                "must be within 3..=31",
            ));
        }
        validate_finite_range(
            "crosshair.confirm_duration_ms",
            self.confirm_duration_ms,
            0.0,
            10_000.0,
        )?;
        validate_finite_range("crosshair.max_age_ms", self.max_age_ms, 1.0, 10_000.0)?;
        validate_finite_range(
            "crosshair.max_offset_px",
            self.max_offset_px,
            0.0,
            f64::from(self.search_size) * 0.5,
        )?;
        validate_finite_range("crosshair.min_similarity", self.min_similarity, 0.0, 1.0)?;
        validate_finite_range(
            "crosshair.max_step_px",
            self.max_step_px,
            0.0,
            self.max_offset_px.max(1.0),
        )?;
        Ok(())
    }
}

const fn default_crosshair_search_size() -> u32 {
    96
}
const fn default_crosshair_sample_hz() -> u32 {
    10
}
const fn default_crosshair_sample_frames() -> usize {
    5
}
const fn default_crosshair_confirm_duration_ms() -> f64 {
    200.0
}
const fn default_crosshair_max_age_ms() -> f64 {
    300.0
}
const fn default_crosshair_max_offset_px() -> f64 {
    20.0
}
const fn default_crosshair_min_similarity() -> f64 {
    0.68
}
const fn default_crosshair_max_step_px() -> f64 {
    2.0
}

#[derive(Clone, Copy, Debug)]
pub struct ProductionAdapterConfig<'a> {
    pub capture: &'a CaptureConfig,
    pub inference: &'a InferenceConfig,
    pub device: &'a DeviceConfig,
    pub pipeline: &'a PipelineRuntimeConfig,
    pub consumers: &'a ConsumerConfig,
    pub limits: &'a LimitsConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionAdapterConfig<'a> {
    pub capture: &'a CaptureConfig,
    pub inference: &'a InferenceConfig,
    pub pipeline: &'a PipelineRuntimeConfig,
    pub consumers: &'a ConsumerConfig,
    pub limits: &'a LimitsConfig,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConsumerConfig {
    #[serde(default)]
    pub preview: bool,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LimitsConfig {
    #[serde(default = "default_stream_fps")]
    pub stream_fps: u32,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            stream_fps: default_stream_fps(),
            extra: BTreeMap::new(),
        }
    }
}

const fn default_stream_fps() -> u32 {
    30
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineRuntimeConfig {
    #[serde(default = "default_freshness_threshold_ms")]
    pub freshness_threshold_ms: f64,
    #[serde(default = "default_projection_fov_x_deg")]
    pub projection_fov_x_deg: f64,
    #[serde(default = "default_projection_counts_per_360")]
    pub projection_counts_per_360: f64,
    #[serde(default = "default_p_response_scale")]
    pub p_response_scale: f64,
    #[serde(default = "default_p_response_boost")]
    pub p_response_boost: f64,
    #[serde(default = "default_response_curve_shape")]
    pub p_response_curve_shape: f64,
    #[serde(default = "default_max_output_counts")]
    pub max_output_x_counts: f64,
    #[serde(default = "default_max_output_counts")]
    pub max_output_y_counts: f64,
    #[serde(default = "default_velocity_history_reset_gap_ms")]
    pub velocity_history_reset_gap_ms: f64,
    #[serde(default = "default_prediction_enabled")]
    pub prediction_enabled: bool,
    #[serde(default = "default_prediction_lead_ms")]
    pub prediction_lead_ms: f64,
    #[serde(default = "default_prediction_cap_px")]
    pub prediction_cap_px: f64,
    #[serde(default = "default_target_fov_radius_px")]
    pub target_fov_radius_px: f64,
    #[serde(default = "default_target_min_confidence")]
    pub target_min_confidence: f32,
    #[serde(default = "default_target_track_max_lost_age_ms")]
    pub target_track_max_lost_age_ms: f64,
    #[serde(default = "default_tracker_max_match_distance")]
    pub tracker_max_match_distance: f64,
    #[serde(default = "default_tracker_position_cost_weight")]
    pub tracker_position_cost_weight: f64,
    #[serde(default = "default_tracker_iou_cost_weight")]
    pub tracker_iou_cost_weight: f64,
    #[serde(default = "default_tracker_scale_cost_weight")]
    pub tracker_scale_cost_weight: f64,
    #[serde(default = "default_tracker_max_size_ratio")]
    pub tracker_max_size_ratio: f64,
    #[serde(default = "default_tracker_max_association_dt_ms")]
    pub tracker_max_association_dt_ms: f64,
    #[serde(default = "default_tracker_kalman_acceleration_noise")]
    pub tracker_kalman_acceleration_noise: f64,
    #[serde(default = "default_tracker_kalman_measurement_noise_x")]
    pub tracker_kalman_measurement_noise_x: f64,
    #[serde(default = "default_tracker_kalman_measurement_noise_y")]
    pub tracker_kalman_measurement_noise_y: f64,
    #[serde(default = "default_tracker_kalman_max_predict_dt_ms")]
    pub tracker_kalman_max_predict_dt_ms: f64,
    #[serde(default = "default_tracker_kalman_max_predict_missing_ms")]
    pub tracker_kalman_max_predict_missing_ms: f64,
    #[serde(default = "default_tracker_kalman_max_predict_steps")]
    pub tracker_kalman_max_predict_steps: u32,
    #[serde(default = "default_tracker_kalman_nis_threshold")]
    pub tracker_kalman_nis_threshold: f64,
    #[serde(default = "default_tracker_kalman_nis_hard_reject")]
    pub tracker_kalman_nis_hard_reject: f64,
    #[serde(default = "default_target_class_priority")]
    pub target_class_priority: String,
    #[serde(default = "default_target_class_filter")]
    pub target_class_filter: String,
    #[serde(default = "default_target_selection_class_ratio")]
    pub target_selection_class_ratio: f64,
    #[serde(default = "default_target_switch_min_preference_advantage")]
    pub target_switch_min_preference_advantage: f64,
    #[serde(default = "default_target_switch_min_continuity_score")]
    pub target_switch_min_continuity_score: f64,
    #[serde(default = "default_target_switch_delay_ms")]
    pub target_switch_delay_ms: f64,
    #[serde(default = "default_target_aim_y_ratio")]
    pub target_aim_y_ratio: f64,
    #[serde(default)]
    pub target_class_aim_y_ratios: String,
    #[serde(default = "default_candidate_max_aspect_ratio")]
    pub candidate_max_aspect_ratio: f64,
    #[serde(
        default = "default_prediction_actuation_delay_ms",
        alias = "actuation_feedback_delay_ms"
    )]
    pub prediction_actuation_delay_ms: f64,
    #[serde(skip)]
    pub(crate) production_fields_explicit: bool,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Default for PipelineRuntimeConfig {
    fn default() -> Self {
        Self {
            freshness_threshold_ms: default_freshness_threshold_ms(),
            projection_fov_x_deg: default_projection_fov_x_deg(),
            projection_counts_per_360: default_projection_counts_per_360(),
            p_response_scale: default_p_response_scale(),
            p_response_boost: default_p_response_boost(),
            p_response_curve_shape: default_response_curve_shape(),
            max_output_x_counts: default_max_output_counts(),
            max_output_y_counts: default_max_output_counts(),
            velocity_history_reset_gap_ms: default_velocity_history_reset_gap_ms(),
            prediction_enabled: default_prediction_enabled(),
            prediction_lead_ms: default_prediction_lead_ms(),
            prediction_cap_px: default_prediction_cap_px(),
            target_fov_radius_px: default_target_fov_radius_px(),
            target_min_confidence: default_target_min_confidence(),
            target_track_max_lost_age_ms: default_target_track_max_lost_age_ms(),
            tracker_max_match_distance: default_tracker_max_match_distance(),
            tracker_position_cost_weight: default_tracker_position_cost_weight(),
            tracker_iou_cost_weight: default_tracker_iou_cost_weight(),
            tracker_scale_cost_weight: default_tracker_scale_cost_weight(),
            tracker_max_size_ratio: default_tracker_max_size_ratio(),
            tracker_max_association_dt_ms: default_tracker_max_association_dt_ms(),
            tracker_kalman_acceleration_noise: default_tracker_kalman_acceleration_noise(),
            tracker_kalman_measurement_noise_x: default_tracker_kalman_measurement_noise_x(),
            tracker_kalman_measurement_noise_y: default_tracker_kalman_measurement_noise_y(),
            tracker_kalman_max_predict_dt_ms: default_tracker_kalman_max_predict_dt_ms(),
            tracker_kalman_max_predict_missing_ms: default_tracker_kalman_max_predict_missing_ms(),
            tracker_kalman_max_predict_steps: default_tracker_kalman_max_predict_steps(),
            tracker_kalman_nis_threshold: default_tracker_kalman_nis_threshold(),
            tracker_kalman_nis_hard_reject: default_tracker_kalman_nis_hard_reject(),
            target_class_priority: default_target_class_priority(),
            target_class_filter: default_target_class_filter(),
            target_selection_class_ratio: default_target_selection_class_ratio(),
            target_switch_min_preference_advantage: default_target_switch_min_preference_advantage(
            ),
            target_switch_min_continuity_score: default_target_switch_min_continuity_score(),
            target_switch_delay_ms: default_target_switch_delay_ms(),
            target_aim_y_ratio: default_target_aim_y_ratio(),
            target_class_aim_y_ratios: String::new(),
            candidate_max_aspect_ratio: default_candidate_max_aspect_ratio(),
            prediction_actuation_delay_ms: default_prediction_actuation_delay_ms(),
            production_fields_explicit: false,
            extra: BTreeMap::new(),
        }
    }
}

impl PipelineRuntimeConfig {
    pub fn continuous_response(&self) -> ContinuousResponseRuntimeConfig {
        ContinuousResponseRuntimeConfig {
            scale: self.p_response_scale,
            boost: self.p_response_boost,
            curve_shape: self.p_response_curve_shape,
        }
    }

    fn validate(&self) -> Result<(), ConfigValidationError> {
        validate_finite_range(
            "pipeline.freshness_threshold_ms",
            self.freshness_threshold_ms,
            1.0,
            1_000.0,
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
            "pipeline.p_response_scale",
            self.p_response_scale,
            0.0,
            100.0,
        )?;
        validate_finite_range(
            "pipeline.p_response_boost",
            self.p_response_boost,
            0.0,
            100.0,
        )?;
        validate_finite_range(
            "pipeline.p_response_curve_shape",
            self.p_response_curve_shape,
            0.5,
            4.0,
        )?;
        validate_finite_range(
            "pipeline.max_output_x_counts",
            self.max_output_x_counts,
            1.0,
            f64::from(i16::MAX),
        )?;
        validate_finite_range(
            "pipeline.max_output_y_counts",
            self.max_output_y_counts,
            1.0,
            f64::from(i16::MAX),
        )?;
        validate_finite_range(
            "pipeline.velocity_history_reset_gap_ms",
            self.velocity_history_reset_gap_ms,
            0.000_001,
            10_000.0,
        )?;
        validate_finite_range(
            "pipeline.prediction_lead_ms",
            self.prediction_lead_ms,
            0.0,
            1_000.0,
        )?;
        validate_finite_range(
            "pipeline.prediction_cap_px",
            self.prediction_cap_px,
            0.0,
            100_000.0,
        )?;
        validate_finite_range(
            "pipeline.target_fov_radius_px",
            self.target_fov_radius_px,
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
        validate_finite_range(
            "pipeline.target_track_max_lost_age_ms",
            self.target_track_max_lost_age_ms,
            1.0,
            10_000.0,
        )?;
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
        validate_finite_range(
            "pipeline.tracker_scale_cost_weight",
            self.tracker_scale_cost_weight,
            0.0,
            100.0,
        )?;
        if self.tracker_position_cost_weight
            + self.tracker_iou_cost_weight
            + self.tracker_scale_cost_weight
            <= 0.0
        {
            return Err(ConfigValidationError::new(
                "pipeline.tracker_position_cost_weight",
                "tracker position, IoU, and scale weights must not all be zero",
            ));
        }
        validate_finite_range(
            "pipeline.tracker_max_size_ratio",
            self.tracker_max_size_ratio,
            1.0,
            100.0,
        )?;
        validate_finite_range(
            "pipeline.tracker_max_association_dt_ms",
            self.tracker_max_association_dt_ms,
            1.0,
            10_000.0,
        )?;
        validate_finite_range(
            "pipeline.tracker_kalman_acceleration_noise",
            self.tracker_kalman_acceleration_noise,
            0.000_001,
            1_000_000.0,
        )?;
        validate_finite_range(
            "pipeline.tracker_kalman_measurement_noise_x",
            self.tracker_kalman_measurement_noise_x,
            0.000_001,
            100_000.0,
        )?;
        validate_finite_range(
            "pipeline.tracker_kalman_measurement_noise_y",
            self.tracker_kalman_measurement_noise_y,
            0.000_001,
            100_000.0,
        )?;
        validate_finite_range(
            "pipeline.tracker_kalman_max_predict_dt_ms",
            self.tracker_kalman_max_predict_dt_ms,
            1.0,
            1_000.0,
        )?;
        validate_finite_range(
            "pipeline.tracker_kalman_max_predict_missing_ms",
            self.tracker_kalman_max_predict_missing_ms,
            1.0,
            10_000.0,
        )?;
        if self.tracker_kalman_max_predict_steps > 120 {
            return Err(ConfigValidationError::new(
                "pipeline.tracker_kalman_max_predict_steps",
                "must be within 0..=120 steps",
            ));
        }
        validate_finite_range(
            "pipeline.tracker_kalman_nis_threshold",
            self.tracker_kalman_nis_threshold,
            0.000_001,
            1_000_000.0,
        )?;
        validate_finite_range(
            "pipeline.tracker_kalman_nis_hard_reject",
            self.tracker_kalman_nis_hard_reject,
            self.tracker_kalman_nis_threshold,
            1_000_000.0,
        )?;
        parse_target_class_priority(&self.target_class_priority)?;
        parse_target_class_filter(&self.target_class_filter)?;
        validate_finite_range(
            "pipeline.target_selection_class_ratio",
            self.target_selection_class_ratio,
            0.0,
            1.0,
        )?;
        validate_finite_range(
            "pipeline.target_switch_min_preference_advantage",
            self.target_switch_min_preference_advantage,
            0.0,
            1.0,
        )?;
        validate_finite_range(
            "pipeline.target_switch_min_continuity_score",
            self.target_switch_min_continuity_score,
            0.0,
            1.0,
        )?;
        validate_finite_range(
            "pipeline.target_switch_delay_ms",
            self.target_switch_delay_ms,
            0.0,
            10_000.0,
        )?;
        validate_finite_range(
            "pipeline.target_aim_y_ratio",
            self.target_aim_y_ratio,
            0.0,
            1.0,
        )?;
        parse_target_class_aim_y_ratios(&self.target_class_aim_y_ratios)?;
        validate_finite_range(
            "pipeline.candidate_max_aspect_ratio",
            self.candidate_max_aspect_ratio,
            1.0,
            100.0,
        )?;
        validate_finite_range(
            "pipeline.prediction_actuation_delay_ms",
            self.prediction_actuation_delay_ms,
            0.0,
            100.0,
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContinuousResponseRuntimeConfig {
    pub scale: f64,
    pub boost: f64,
    pub curve_shape: f64,
}

pub fn parse_target_class_priority(value: &str) -> Result<Vec<u32>, ConfigValidationError> {
    let mut classes = Vec::new();
    for item in value.split(',') {
        let class_id = item.trim().parse::<u32>().map_err(|_| {
            ConfigValidationError::new(
                "pipeline.target_class_priority",
                "must be a comma-separated list of unique class ids",
            )
        })?;
        if classes.contains(&class_id) {
            return Err(ConfigValidationError::new(
                "pipeline.target_class_priority",
                "must not contain duplicate class ids",
            ));
        }
        classes.push(class_id);
    }
    if classes.is_empty() {
        return Err(ConfigValidationError::new(
            "pipeline.target_class_priority",
            "must contain at least one class id",
        ));
    }
    Ok(classes)
}

pub fn parse_target_class_filter(
    value: &str,
) -> Result<Option<BTreeSet<u32>>, ConfigValidationError> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized == "all" {
        return Ok(None);
    }
    if normalized == "none" {
        return Ok(Some(BTreeSet::new()));
    }
    let mut classes = BTreeSet::new();
    for item in normalized.split(',') {
        let class_id = item.trim().parse::<u32>().map_err(|_| {
            ConfigValidationError::new(
                "pipeline.target_class_filter",
                "must be all, none, or a comma-separated list of unique unsigned class ids",
            )
        })?;
        if !classes.insert(class_id) {
            return Err(ConfigValidationError::new(
                "pipeline.target_class_filter",
                "must be all, none, or a comma-separated list of unique unsigned class ids",
            ));
        }
    }
    if classes.is_empty() {
        return Err(ConfigValidationError::new(
            "pipeline.target_class_filter",
            "must be all, none, or a comma-separated list of unique unsigned class ids",
        ));
    }
    Ok(Some(classes))
}

pub fn parse_target_class_aim_y_ratios(
    value: &str,
) -> Result<BTreeMap<u32, f64>, ConfigValidationError> {
    let mut ratios = BTreeMap::new();
    if value.trim().is_empty() {
        return Ok(ratios);
    }
    for item in value.split(',') {
        let (class_id, ratio) = item.trim().split_once(':').ok_or_else(|| {
            ConfigValidationError::new(
                "pipeline.target_class_aim_y_ratios",
                "must use class_id:ratio pairs separated by commas",
            )
        })?;
        let class_id = class_id.trim().parse::<u32>().map_err(|_| {
            ConfigValidationError::new(
                "pipeline.target_class_aim_y_ratios",
                "class ids must be unsigned integers",
            )
        })?;
        let ratio = ratio.trim().parse::<f64>().map_err(|_| {
            ConfigValidationError::new(
                "pipeline.target_class_aim_y_ratios",
                "ratios must be finite numbers within 0..=1",
            )
        })?;
        if !ratio.is_finite()
            || !(0.0..=1.0).contains(&ratio)
            || ratios.insert(class_id, ratio).is_some()
        {
            return Err(ConfigValidationError::new(
                "pipeline.target_class_aim_y_ratios",
                "class ids must be unique and ratios must be finite within 0..=1",
            ));
        }
    }
    Ok(ratios)
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

const fn default_projection_fov_x_deg() -> f64 {
    105.0
}

const fn default_projection_counts_per_360() -> f64 {
    9_980.0
}

const fn default_response_curve_shape() -> f64 {
    1.0
}

const fn default_p_response_scale() -> f64 {
    0.20
}

const fn default_p_response_boost() -> f64 {
    0.50
}

const fn default_velocity_history_reset_gap_ms() -> f64 {
    80.0
}

const fn default_prediction_enabled() -> bool {
    true
}

const fn default_prediction_lead_ms() -> f64 {
    16.0
}

const fn default_max_output_counts() -> f64 {
    127.0
}

const fn default_prediction_cap_px() -> f64 {
    10.0
}

const fn default_target_fov_radius_px() -> f64 {
    180.0
}

const fn default_target_min_confidence() -> f32 {
    0.5
}

const fn default_target_track_max_lost_age_ms() -> f64 {
    120.0
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

const fn default_tracker_scale_cost_weight() -> f64 {
    0.15
}

const fn default_tracker_max_size_ratio() -> f64 {
    2.5
}

const fn default_tracker_max_association_dt_ms() -> f64 {
    150.0
}

fn default_tracker_kalman_acceleration_noise() -> f64 {
    KalmanConfig::default().acceleration_noise
}

fn default_tracker_kalman_measurement_noise_x() -> f64 {
    KalmanConfig::default().measurement_noise_x
}

fn default_tracker_kalman_measurement_noise_y() -> f64 {
    KalmanConfig::default().measurement_noise_y
}

fn default_tracker_kalman_max_predict_dt_ms() -> f64 {
    KalmanConfig::default().max_predict_dt_ms
}

fn default_tracker_kalman_max_predict_missing_ms() -> f64 {
    KalmanConfig::default().max_predict_missing_ms
}

fn default_tracker_kalman_max_predict_steps() -> u32 {
    KalmanConfig::default().max_predict_steps
}

fn default_tracker_kalman_nis_threshold() -> f64 {
    KalmanConfig::default().nis_threshold
}

fn default_tracker_kalman_nis_hard_reject() -> f64 {
    KalmanConfig::default().nis_hard_reject
}

fn default_target_class_priority() -> String {
    "0,1".to_owned()
}

fn default_target_class_filter() -> String {
    "all".to_owned()
}

const fn default_target_selection_class_ratio() -> f64 {
    0.35
}

const fn default_target_switch_min_preference_advantage() -> f64 {
    0.08
}

const fn default_target_switch_min_continuity_score() -> f64 {
    0.70
}

const fn default_target_switch_delay_ms() -> f64 {
    50.0
}

const fn default_target_aim_y_ratio() -> f64 {
    0.22
}

const fn default_candidate_max_aspect_ratio() -> f64 {
    6.0
}

const fn default_prediction_actuation_delay_ms() -> f64 {
    4.0
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
    #[serde(default, skip_serializing)]
    pub backend: DeepStreamBackend,
    #[serde(default, skip_serializing)]
    pub memory: CaptureMemory,
    #[serde(default)]
    pub preference: CapturePreference,
    #[serde(default = "default_true", skip_serializing)]
    pub latest_only: bool,
    #[serde(default = "default_one_u32", skip_serializing)]
    pub appsink_max_buffers: u32,
    #[serde(default, skip_serializing)]
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
    pub extra: BTreeMap<String, Value>,
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
            extra: BTreeMap::new(),
        }
    }
}

impl CaptureConfig {
    pub fn validate_runtime_plan(&self) -> Result<(), ConfigValidationError> {
        self.validate()
    }

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
    #[serde(default, skip_serializing)]
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
    #[serde(default = "default_deepstream_io_mode", skip_serializing)]
    pub deepstream_io_mode: i32,
    #[serde(default, skip_serializing)]
    pub deepstream_batched_push_timeout_us: i64,
    #[serde(default = "default_inference_component_id", skip_serializing)]
    pub deepstream_component_id: i32,
    #[serde(default, skip_serializing)]
    pub deepstream_source_id: u32,
    #[serde(default = "default_probe_element", skip_serializing)]
    pub deepstream_probe_element: String,
    #[serde(default = "default_probe_pad", skip_serializing)]
    pub deepstream_probe_pad: String,
    #[serde(default = "default_nvinfer_config", skip_serializing)]
    pub deepstream_nvinfer_config: PathBuf,
    #[serde(default, skip_serializing)]
    pub model_width: u32,
    #[serde(default, skip_serializing)]
    pub model_height: u32,
    #[serde(default = "default_deepstream_startup_timeout_ms")]
    pub deepstream_startup_timeout_ms: u64,
    #[serde(default = "default_deepstream_shutdown_timeout_ms")]
    pub deepstream_shutdown_timeout_ms: u64,
    #[serde(default, skip_serializing)]
    pub input_source: InferenceInputSource,
    #[serde(skip)]
    pub(crate) production_fields_explicit: bool,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
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
            extra: BTreeMap::new(),
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
        if self
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
    #[serde(default = "default_kmnet_connect_timeout_ms")]
    pub connect_timeout_ms: u64,
    #[serde(default = "default_kmnet_send_timeout_ms")]
    pub send_timeout_ms: u64,
    #[serde(default = "default_kmnet_monitor_timeout_ms")]
    pub monitor_timeout_ms: u64,
    #[serde(default = "default_kmnet_trigger_poll_interval_ms")]
    pub trigger_poll_interval_ms: u64,
    #[serde(skip)]
    pub(crate) production_fields_explicit: bool,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
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
            connect_timeout_ms: default_kmnet_connect_timeout_ms(),
            send_timeout_ms: default_kmnet_send_timeout_ms(),
            monitor_timeout_ms: default_kmnet_monitor_timeout_ms(),
            trigger_poll_interval_ms: default_kmnet_trigger_poll_interval_ms(),
            production_fields_explicit: false,
            extra: BTreeMap::new(),
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
        if self.auto_connect
            && (self.uuid.trim().len() != 8
                || !self
                    .uuid
                    .trim()
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()))
        {
            return Err(ConfigValidationError::new(
                "hardware.uuid",
                "must contain exactly eight hexadecimal digits",
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
        if self.auto_connect && self.monitor_timeout_ms == 0 {
            return Err(ConfigValidationError::new(
                "hardware.monitor_timeout_ms",
                "must be non-zero for native_udp",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceBackend {
    #[default]
    #[serde(alias = "python_host")]
    NativeUdp,
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
    pub extra: BTreeMap<String, Value>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_server_host(),
            port: default_server_port(),
            control_socket: default_control_socket(),
            extra: BTreeMap::new(),
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
    pub extra: BTreeMap<String, Value>,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            enabled: default_replay_enabled(),
            frame_interval_ms: default_frame_interval_ms(),
            output_gate_open: false,
            extra: BTreeMap::new(),
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
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Default for PathConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            model_dir: default_model_dir(),
            database: default_database(),
            license: default_license(),
            extra: BTreeMap::new(),
        }
    }
}

const fn default_schema_version() -> u32 {
    CURRENT_SCHEMA_VERSION
}

fn default_server_host() -> String {
    "0.0.0.0".to_owned()
}

const fn default_server_port() -> u16 {
    5174
}

fn default_control_socket() -> PathBuf {
    PathBuf::from("run/novasightd.sock")
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
    PathBuf::from("auto")
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
    String::new()
}

const fn default_kmnet_port() -> u16 {
    8888
}

fn default_kmnet_uuid() -> String {
    String::new()
}

const fn default_kmnet_monitor_port() -> u16 {
    5001
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_association_requires_at_least_one_real_cost_signal() {
        let config = PipelineRuntimeConfig {
            tracker_position_cost_weight: 0.0,
            tracker_iou_cost_weight: 0.0,
            tracker_scale_cost_weight: 0.0,
            ..PipelineRuntimeConfig::default()
        };
        let error = config.validate().expect_err("zero association weights");
        assert_eq!(error.field, "pipeline.tracker_position_cost_weight");
    }

    #[test]
    fn tracker_kalman_prediction_steps_are_bounded() {
        let config = PipelineRuntimeConfig {
            tracker_kalman_max_predict_steps: 121,
            ..PipelineRuntimeConfig::default()
        };
        let error = config
            .validate()
            .expect_err("unbounded kalman prediction steps");
        assert_eq!(error.field, "pipeline.tracker_kalman_max_predict_steps");
    }

    #[test]
    fn continuous_response_uses_canonical_fields() {
        let config = PipelineRuntimeConfig {
            p_response_scale: 0.42,
            p_response_boost: 0.60,
            p_response_curve_shape: 1.50,
            ..PipelineRuntimeConfig::default()
        };

        let response = config.continuous_response();

        assert_eq!(response.scale, 0.42);
        assert_eq!(response.boost, 0.60);
        assert_eq!(response.curve_shape, 1.50);
    }

    #[test]
    fn target_priority_rejects_duplicates_before_runtime_composition() {
        let config = PipelineRuntimeConfig {
            target_class_priority: "0,1,0".to_owned(),
            ..PipelineRuntimeConfig::default()
        };
        let error = config.validate().expect_err("duplicate class priority");
        assert_eq!(error.field, "pipeline.target_class_priority");
    }

    #[test]
    fn target_class_filter_preserves_all_none_and_explicit_allowlist() {
        assert_eq!(parse_target_class_filter("all").unwrap(), None);
        assert!(
            parse_target_class_filter("none")
                .unwrap()
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            parse_target_class_filter("1, 0, 3")
                .unwrap()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![0, 1, 3]
        );
        assert!(parse_target_class_filter("").is_err());
        assert!(parse_target_class_filter("0,0").is_err());
        assert_eq!(
            parse_target_class_filter("256")
                .unwrap()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![256]
        );
    }

    #[test]
    fn target_selection_class_ratio_is_bounded() {
        let config = PipelineRuntimeConfig {
            target_selection_class_ratio: 1.1,
            ..PipelineRuntimeConfig::default()
        };
        let error = config.validate().expect_err("class ratio above one");
        assert_eq!(error.field, "pipeline.target_selection_class_ratio");
    }

    #[test]
    fn class_aim_ratios_parse_to_typed_bounded_values() {
        let parsed = parse_target_class_aim_y_ratios("0:0.22, 1:0.30").unwrap();
        assert_eq!(parsed.get(&0), Some(&0.22));
        assert_eq!(parsed.get(&1), Some(&0.30));
        assert!(parse_target_class_aim_y_ratios("0:1.1").is_err());
        assert!(parse_target_class_aim_y_ratios("0:0.2,0:0.3").is_err());
    }

    #[test]
    fn preview_configuration_is_typed_and_rejects_zero_fps() {
        let mut config: AppConfig =
            serde_yaml::from_str("consumers:\n  preview: true\nlimits:\n  stream_fps: 24\n")
                .unwrap();
        assert!(config.consumers.preview);
        assert_eq!(config.limits.stream_fps, 24);
        config.validate_configured_adapters().unwrap();

        config.limits.stream_fps = 0;
        let error = config.validate_configured_adapters().unwrap_err();
        assert_eq!(error.field, "limits.stream_fps");
    }

    #[test]
    fn crosshair_configuration_is_typed_and_bounded() {
        let mut config: AppConfig = serde_yaml::from_str(
            "crosshair:\n  enabled: true\n  use_for_control: true\n  search_size: 96\n  sample_hz: 10\n  sample_frames: 5\n  max_offset_px: 20\n",
        )
        .unwrap();
        assert!(config.crosshair.enabled);
        assert!(config.crosshair.use_for_control);
        config.validate_configured_adapters().unwrap();

        config.crosshair.max_offset_px = 49.0;
        let error = config.validate_configured_adapters().unwrap_err();
        assert_eq!(error.field, "crosshair.max_offset_px");
    }

    #[test]
    fn recoil_configuration_is_typed_and_bounded() {
        let mut config: AppConfig = serde_yaml::from_str(
            "control:\n  recoil:\n    enabled: true\n    interval_ms: 20\n    fire_delay_ms: 40\n    y_counts: 3\n",
        )
        .unwrap();
        assert!(config.control.recoil.enabled);
        assert_eq!(config.control.recoil.interval_ms, 20);
        assert_eq!(config.control.recoil.fire_delay_ms, 40);
        assert_eq!(config.control.recoil.y_counts, 3);
        config.validate_configured_adapters().unwrap();

        config.control.recoil.y_counts = 0;
        let error = config.validate_configured_adapters().unwrap_err();
        assert_eq!(error.field, "control.recoil.y_counts");
    }

    #[test]
    fn output_gate_is_typed_and_defaults_closed_for_safe_commissioning() {
        let defaulted: AppConfig = serde_yaml::from_str("{}\n").unwrap();
        assert!(!defaulted.control.output_enabled);

        let paused: AppConfig =
            serde_yaml::from_str("control:\n  output_enabled: false\n").unwrap();
        assert!(!paused.control.output_enabled);
        assert!(!paused.control.extra.contains_key("output_enabled"));
    }

    #[test]
    fn configured_output_gate_rejects_an_explicitly_uncommissioned_device() {
        let config: AppConfig = serde_yaml::from_str(
            "control:\n  output_enabled: true\nhardware:\n  auto_connect: false\n",
        )
        .unwrap();

        let error = config.validate_configured_adapters().unwrap_err();
        assert_eq!(error.field, "control.output_enabled");
        assert!(error.message.contains("hardware.auto_connect"));
    }

    #[test]
    fn configured_output_gate_requires_a_hardware_section() {
        let config: AppConfig = serde_yaml::from_str("control:\n  output_enabled: true\n").unwrap();

        let error = config.validate_configured_adapters().unwrap_err();
        assert_eq!(error.field, "control.output_enabled");
        assert!(error.message.contains("hardware"));
    }

    #[test]
    fn device_auto_connect_does_not_invent_missing_identity() {
        let config: AppConfig = serde_yaml::from_str("hardware:\n  auto_connect: true\n").unwrap();

        let error = config.validate_configured_adapters().unwrap_err();
        assert_eq!(error.field, "hardware.host");
    }

    #[test]
    fn device_auto_connect_accepts_valid_vendor_uuid_values_without_business_blacklists() {
        for uuid in ["12345678", "00000000"] {
            let config: AppConfig = serde_yaml::from_str(&format!(
                "hardware:\n  auto_connect: true\n  host: 192.168.2.188\n  uuid: '{uuid}'\n"
            ))
            .unwrap();

            config.validate_configured_adapters().unwrap();
        }
    }

    #[test]
    fn device_auto_connect_requires_the_vendor_uuid_contract() {
        let config: AppConfig = serde_yaml::from_str(
            "hardware:\n  auto_connect: true\n  host: 192.168.2.188\n  uuid: test-box\n",
        )
        .unwrap();

        let error = config.validate_configured_adapters().unwrap_err();
        assert_eq!(error.field, "hardware.uuid");
        assert!(error.message.contains("eight hexadecimal digits"));
    }
}
