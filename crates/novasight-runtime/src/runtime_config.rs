use novasight_core::controller::DualPhaseConfig;
use novasight_core::controller::recoil::RecoilConfig;
use novasight_core::tracking::{KalmanConfig, TargetingConfig};
use novasight_pipeline::{PipelineConfig, TriggerMode};
use novasight_store::config::{
    AppConfig, parse_target_class_aim_y_ratios, parse_target_class_filter,
    parse_target_class_priority,
};

/// Compose the epoch-scoped control pipeline from the canonical application
/// configuration. Both daemon bootstrap and stopped-runtime reload use this
/// function so a saved setting cannot mean something different after restart.
pub fn compose_pipeline_config(
    config: &AppConfig,
    trigger_poll_interval_ms: Option<u64>,
) -> Result<PipelineConfig, String> {
    let adapters = config
        .require_vision_adapters()
        .map_err(|error| error.to_string())?;
    if !adapters.inference.enabled {
        return Err("inference.enabled must be true for the live runtime".to_owned());
    }
    Ok(PipelineConfig {
        targeting: TargetingConfig {
            target_fov_radius_px: adapters.pipeline.target_fov_radius_px,
            min_confidence: adapters.pipeline.target_min_confidence,
            track_max_age: adapters.pipeline.target_track_max_age,
            track_max_lost_age_ms: adapters.pipeline.target_track_max_lost_age_ms,
            tracker_max_match_distance: adapters.pipeline.tracker_max_match_distance,
            tracker_position_cost_weight: adapters.pipeline.tracker_position_cost_weight,
            tracker_iou_cost_weight: adapters.pipeline.tracker_iou_cost_weight,
            tracker_scale_cost_weight: adapters.pipeline.tracker_scale_cost_weight,
            tracker_max_size_ratio: adapters.pipeline.tracker_max_size_ratio,
            tracker_max_association_dt_ms: adapters.pipeline.tracker_max_association_dt_ms,
            kalman: KalmanConfig {
                acceleration_noise: adapters.pipeline.tracker_kalman_acceleration_noise,
                measurement_noise_x: adapters.pipeline.tracker_kalman_measurement_noise_x,
                measurement_noise_y: adapters.pipeline.tracker_kalman_measurement_noise_y,
                max_predict_dt_ms: adapters.pipeline.tracker_kalman_max_predict_dt_ms,
                max_predict_missing_ms: adapters.pipeline.tracker_kalman_max_predict_missing_ms,
                max_predict_steps: adapters.pipeline.tracker_kalman_max_predict_steps,
                nis_threshold: adapters.pipeline.tracker_kalman_nis_threshold,
                nis_hard_reject: adapters.pipeline.tracker_kalman_nis_hard_reject,
                ..KalmanConfig::default()
            },
            class_priority: parse_target_class_priority(&adapters.pipeline.target_class_priority)
                .map_err(|error| error.to_string())?,
            allowed_class_ids: parse_target_class_filter(&adapters.pipeline.target_class_filter)
                .map_err(|error| error.to_string())?,
            selection_class_ratio: adapters.pipeline.target_selection_class_ratio,
            switch_min_preference_advantage: adapters
                .pipeline
                .target_switch_min_preference_advantage,
            switch_min_continuity_score: adapters.pipeline.target_switch_min_continuity_score,
            switch_delay_ms: adapters.pipeline.target_switch_delay_ms,
            aim_y_ratio: adapters.pipeline.target_aim_y_ratio,
            class_aim_y_ratios: parse_target_class_aim_y_ratios(
                &adapters.pipeline.target_class_aim_y_ratios,
            )
            .map_err(|error| error.to_string())?,
            candidate_max_aspect_ratio: adapters.pipeline.candidate_max_aspect_ratio,
        },
        control: DualPhaseConfig {
            freshness_threshold_ms: adapters.pipeline.freshness_threshold_ms,
            near_threshold_px: adapters.pipeline.near_threshold_px,
            projection_fov_x_deg: adapters.pipeline.projection_fov_x_deg,
            projection_counts_per_360: adapters.pipeline.projection_counts_per_360,
            atan_scale_counts: adapters.pipeline.atan_scale_counts,
            far_kp: adapters.pipeline.far_kp,
            far_max_counts_per_update: adapters.pipeline.far_max_counts_per_update,
            near_kp: adapters.pipeline.near_kp,
            near_max_counts_per_update: adapters.pipeline.near_max_counts_per_update,
            arrival_radius_counts: adapters.pipeline.arrival_radius_counts,
            velocity_smoothing_frames: adapters.pipeline.velocity_smoothing_frames,
            velocity_history_reset_gap_ms: adapters.pipeline.velocity_history_reset_gap_ms,
            velocity_spread_base_px_ms: adapters.pipeline.velocity_spread_base_px_ms,
            velocity_spread_relative: adapters.pipeline.velocity_spread_relative,
            velocity_change_base_px_ms: adapters.pipeline.velocity_change_base_px_ms,
            velocity_change_relative: adapters.pipeline.velocity_change_relative,
            prediction_enabled: adapters.pipeline.prediction_enabled,
            prediction_actuation_delay_ms: adapters.pipeline.actuation_feedback_delay_ms,
            prediction_lead_frames: adapters.pipeline.prediction_lead_frames,
            prediction_far_absolute_cap_px: adapters.pipeline.prediction_far_absolute_cap_px,
            prediction_far_base_cap_px: adapters.pipeline.prediction_far_base_cap_px,
            prediction_far_relative_cap: adapters.pipeline.prediction_far_relative_cap,
            prediction_near_absolute_cap_px: adapters.pipeline.prediction_near_absolute_cap_px,
            prediction_near_base_cap_px: adapters.pipeline.prediction_near_base_cap_px,
            prediction_near_relative_cap: adapters.pipeline.prediction_near_relative_cap,
            source_width: adapters.capture.width,
            roi_width: adapters.capture.roi_width,
            roi_height: adapters.capture.roi_height,
            observation_width: 0,
            observation_height: 0,
            residual_cap: adapters.pipeline.residual_cap,
        },
        actuation_feedback_delay_ns: (adapters.pipeline.actuation_feedback_delay_ms * 1_000_000.0)
            .round() as u64,
        trigger_poll_interval_ms,
        trigger_mode: match config.control.trigger_mode {
            novasight_store::config::TriggerMode::Always => TriggerMode::Always,
            novasight_store::config::TriggerMode::Hardware => TriggerMode::Hardware,
        },
        recoil: RecoilConfig {
            enabled: config.control.recoil.enabled,
            require_target: config.control.recoil.require_target,
            interval_ms: config.control.recoil.interval_ms,
            y_counts: config.control.recoil.y_counts,
        },
        ..PipelineConfig::default()
    })
}
