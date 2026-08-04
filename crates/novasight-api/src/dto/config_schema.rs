use novasight_core::controller::{
    ATAN_RESPONSE_MOTION_BOOST_FRACTION, ATAN_RESPONSE_STATIC_BOOST_FRACTION,
    DEFAULT_ATAN_SCALE_COUNTS,
};
use novasight_runtime::AppConfig;
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ConfigSchemaResponse {
    version: u32,
    algorithm: ConfigAlgorithmSchema,
    values: Value,
    sections: Vec<ConfigSectionSchema>,
}

#[derive(Clone, Debug, Serialize)]
struct ConfigAlgorithmSchema {
    id: &'static str,
    label: &'static str,
    response: ConfigAlgorithmResponseSchema,
    prediction: ConfigAlgorithmPredictionSchema,
}

#[derive(Clone, Debug, Serialize)]
struct ConfigAlgorithmResponseSchema {
    formula: &'static str,
    radial_multiplier_formula: &'static str,
    atan_scale_counts: f64,
    static_acquisition_boost_fraction: f64,
    motion_boost_fraction: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ConfigAlgorithmPredictionSchema {
    model: &'static str,
    aim_history_points: u32,
    velocity_segments: u32,
}

#[derive(Clone, Debug, Serialize)]
struct ConfigSectionSchema {
    id: &'static str,
    label: &'static str,
    fields: Vec<ConfigFieldSchema>,
}

#[derive(Clone, Debug, Serialize)]
struct ConfigFieldSchema {
    path: &'static str,
    label: &'static str,
    #[serde(rename = "type")]
    field_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<&'static [&'static str]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'static str>,
    restart_required: bool,
}

impl ConfigSchemaResponse {
    pub(crate) fn new(config: &AppConfig) -> Self {
        Self {
            version: config.schema_version,
            algorithm: ConfigAlgorithmSchema {
                id: "continuous_atan_predictive_v1",
                label: "连续 Atan 控制",
                response: ConfigAlgorithmResponseSchema {
                    formula: "u = K_base * R(r, motion_strength) * S * atan(e_pred / S)",
                    radial_multiplier_formula: "R = 1 + B * (static + motion * motion_strength) * (1 - exp(-(r ^ gamma)))",
                    atan_scale_counts: DEFAULT_ATAN_SCALE_COUNTS,
                    static_acquisition_boost_fraction: ATAN_RESPONSE_STATIC_BOOST_FRACTION,
                    motion_boost_fraction: ATAN_RESPONSE_MOTION_BOOST_FRACTION,
                },
                prediction: ConfigAlgorithmPredictionSchema {
                    model: "motion-gated four-point velocity",
                    aim_history_points: 4,
                    velocity_segments: 3,
                },
            },
            values: serde_json::to_value(config)
                .expect("validated AppConfig must have a JSON representation"),
            sections: vec![
                section(
                    "server",
                    "服务控制面",
                    vec![
                        string("server.host", "监听地址"),
                        integer("server.port", "监听端口", 1.0, 65_535.0, None),
                        string("server.control_socket", "本地控制套接字"),
                    ],
                ),
                section(
                    "replay",
                    "安全回放",
                    vec![
                        boolean("replay.enabled", "启用回放模式"),
                        integer(
                            "replay.frame_interval_ms",
                            "回放帧间隔",
                            1.0,
                            u32::MAX as f64,
                            Some("ms"),
                        ),
                        boolean("replay.output_gate_open", "允许回放输出"),
                    ],
                ),
                section(
                    "consumers",
                    "预览消费者",
                    vec![boolean("consumers.preview", "启用硬件 JPEG 预览")],
                ),
                section(
                    "limits",
                    "流量限制",
                    vec![integer(
                        "limits.stream_fps",
                        "预览帧率上限",
                        1.0,
                        u32::MAX as f64,
                        Some("Hz"),
                    )],
                ),
                section(
                    "crosshair",
                    "视觉准星",
                    vec![
                        boolean("crosshair.enabled", "启用视觉准星观察"),
                        boolean("crosshair.use_for_control", "使用视觉准星作为控制原点"),
                        integer(
                            "crosshair.search_size",
                            "中心搜索区域",
                            32.0,
                            640.0,
                            Some("px"),
                        ),
                        integer("crosshair.sample_hz", "观察采样率", 1.0, 60.0, Some("Hz")),
                        integer(
                            "crosshair.sample_frames",
                            "学习样本帧数",
                            3.0,
                            31.0,
                            Some("frame"),
                        ),
                        float(
                            "crosshair.confirm_duration_ms",
                            "稳定确认时长",
                            0.0,
                            10_000.0,
                            Some("ms"),
                        ),
                        float(
                            "crosshair.max_age_ms",
                            "观察新鲜度上限",
                            1.0,
                            10_000.0,
                            Some("ms"),
                        ),
                        float(
                            "crosshair.max_offset_px",
                            "最大搜索偏移",
                            0.0,
                            320.0,
                            Some("px"),
                        ),
                        float("crosshair.min_similarity", "最低模板相似度", 0.0, 1.0, None),
                        float(
                            "crosshair.max_step_px",
                            "确认点单帧最大变化",
                            0.0,
                            320.0,
                            Some("px"),
                        ),
                    ],
                ),
                section(
                    "control",
                    "控制输出",
                    vec![
                        hot_boolean("control.output_enabled", "允许设备位移输出"),
                        hot_select(
                            "control.trigger_mode",
                            "输出触发方式",
                            &["always", "hardware"],
                        ),
                    ],
                ),
                runtime_section(
                    "control.recoil",
                    "独立压枪",
                    vec![
                        boolean("control.recoil.enabled", "启用独立 Y 轴压枪"),
                        boolean("control.recoil.require_target", "只在存在目标时压枪"),
                        integer(
                            "control.recoil.interval_ms",
                            "压枪叠加间隔",
                            1.0,
                            5_000.0,
                            Some("ms"),
                        ),
                        integer(
                            "control.recoil.y_counts",
                            "每次叠加 +Y",
                            1.0,
                            i16::MAX as f64,
                            Some("counts"),
                        ),
                    ],
                ),
                runtime_section(
                    "pipeline",
                    "Rust 实时控制",
                    vec![
                        float(
                            "pipeline.freshness_threshold_ms",
                            "观测新鲜度上限",
                            1.0,
                            1_000.0,
                            Some("ms"),
                        ),
                        float(
                            "pipeline.projection_fov_x_deg",
                            "水平视场角",
                            0.000_001,
                            179.999_999,
                            Some("deg"),
                        ),
                        float(
                            "pipeline.projection_counts_per_360",
                            "水平一周鼠标计数",
                            0.000_001,
                            1_000_000.0,
                            Some("count"),
                        ),
                        float(
                            "pipeline.p_response_scale",
                            "基础响应 K_base",
                            0.0,
                            100.0,
                            None,
                        ),
                        float("pipeline.p_response_boost", "动态增强 B", 0.0, 100.0, None),
                        float(
                            "pipeline.p_response_curve_shape",
                            "过渡形状 gamma",
                            0.5,
                            4.0,
                            None,
                        ),
                        float(
                            "pipeline.max_counts_per_update",
                            "单次计数上限",
                            1.0,
                            i16::MAX as f64,
                            Some("count"),
                        ),
                        float(
                            "pipeline.arrival_radius_counts",
                            "FOV 投影后的到位半径",
                            0.5,
                            1_000.0,
                            Some("count"),
                        ),
                        boolean("pipeline.prediction_enabled", "启用目标速度预测"),
                        float(
                            "pipeline.velocity_history_reset_gap_ms",
                            "预测历史重置间隔",
                            0.000_001,
                            10_000.0,
                            Some("ms"),
                        ),
                        float(
                            "pipeline.velocity_spread_base_px_ms",
                            "二维速度离散基础容差",
                            0.000_001,
                            10_000.0,
                            Some("px/ms"),
                        ),
                        float(
                            "pipeline.velocity_spread_relative",
                            "二维速度离散相对容差",
                            0.0,
                            100.0,
                            None,
                        ),
                        float(
                            "pipeline.prediction_lead_ms",
                            "目标速度预测额外提前量",
                            0.0,
                            1_000.0,
                            Some("ms"),
                        ),
                        float(
                            "pipeline.prediction_cap_px",
                            "预测位移上限",
                            0.0,
                            100_000.0,
                            Some("px"),
                        ),
                        float(
                            "pipeline.residual_cap",
                            "量化残差上限",
                            0.0,
                            1.0,
                            Some("count"),
                        ),
                        float(
                            "pipeline.target_fov_radius_px",
                            "目标选择半径",
                            0.000_001,
                            100_000.0,
                            Some("px"),
                        ),
                        float(
                            "pipeline.target_min_confidence",
                            "控制目标最低置信度",
                            0.0,
                            1.0,
                            None,
                        ),
                        integer(
                            "pipeline.target_track_max_age",
                            "无时间戳回放漏检上限",
                            1.0,
                            120.0,
                            Some("frame"),
                        ),
                        float(
                            "pipeline.target_track_max_lost_age_ms",
                            "丢失轨迹最长保留时间",
                            1.0,
                            10_000.0,
                            Some("ms"),
                        ),
                        float(
                            "pipeline.tracker_max_match_distance",
                            "跟踪最大匹配距离",
                            0.000_001,
                            100.0,
                            Some("target-height"),
                        ),
                        float(
                            "pipeline.tracker_position_cost_weight",
                            "跟踪位置代价权重",
                            0.0,
                            100.0,
                            None,
                        ),
                        float(
                            "pipeline.tracker_iou_cost_weight",
                            "跟踪 IoU 代价权重",
                            0.0,
                            100.0,
                            None,
                        ),
                        float(
                            "pipeline.tracker_scale_cost_weight",
                            "跟踪尺度代价权重",
                            0.0,
                            100.0,
                            None,
                        ),
                        float(
                            "pipeline.tracker_max_size_ratio",
                            "跟踪最大尺寸变化倍数",
                            1.0,
                            100.0,
                            Some("ratio"),
                        ),
                        float(
                            "pipeline.tracker_max_association_dt_ms",
                            "跟踪最大关联时间间隔",
                            1.0,
                            10_000.0,
                            Some("ms"),
                        ),
                        float(
                            "pipeline.tracker_kalman_acceleration_noise",
                            "卡尔曼运动响应噪声",
                            0.000_001,
                            1_000_000.0,
                            None,
                        ),
                        float(
                            "pipeline.tracker_kalman_measurement_noise_x",
                            "卡尔曼 X 轴观测噪声",
                            0.000_001,
                            100_000.0,
                            Some("px2"),
                        ),
                        float(
                            "pipeline.tracker_kalman_measurement_noise_y",
                            "卡尔曼 Y 轴观测噪声",
                            0.000_001,
                            100_000.0,
                            Some("px2"),
                        ),
                        float(
                            "pipeline.tracker_kalman_max_predict_dt_ms",
                            "卡尔曼单步预测上限",
                            1.0,
                            1_000.0,
                            Some("ms"),
                        ),
                        float(
                            "pipeline.tracker_kalman_max_predict_missing_ms",
                            "卡尔曼丢失预测窗口",
                            1.0,
                            10_000.0,
                            Some("ms"),
                        ),
                        integer(
                            "pipeline.tracker_kalman_max_predict_steps",
                            "卡尔曼连续预测步数",
                            0.0,
                            120.0,
                            Some("step"),
                        ),
                        float(
                            "pipeline.tracker_kalman_nis_threshold",
                            "卡尔曼可信 NIS 阈值",
                            0.000_001,
                            1_000_000.0,
                            None,
                        ),
                        float(
                            "pipeline.tracker_kalman_nis_hard_reject",
                            "卡尔曼 NIS 硬拒绝阈值",
                            0.000_001,
                            1_000_000.0,
                            None,
                        ),
                        string("pipeline.target_class_priority", "目标类别优先级"),
                        string("pipeline.target_class_filter", "参与目标选择的类别"),
                        float(
                            "pipeline.target_selection_class_ratio",
                            "目标类别偏好比例",
                            0.0,
                            1.0,
                            None,
                        ),
                        float(
                            "pipeline.target_switch_min_preference_advantage",
                            "目标切换最小评分优势",
                            0.0,
                            1.0,
                            None,
                        ),
                        float(
                            "pipeline.target_switch_min_continuity_score",
                            "目标切换最小连续性",
                            0.0,
                            1.0,
                            None,
                        ),
                        float(
                            "pipeline.target_switch_delay_ms",
                            "目标切换确认延迟",
                            0.0,
                            10_000.0,
                            Some("ms"),
                        ),
                        float(
                            "pipeline.target_aim_y_ratio",
                            "默认垂直瞄点比例",
                            0.0,
                            1.0,
                            Some("bbox-height"),
                        ),
                        string("pipeline.target_class_aim_y_ratios", "按类别垂直瞄点比例"),
                        float(
                            "pipeline.candidate_max_aspect_ratio",
                            "候选框最大宽高比",
                            1.0,
                            100.0,
                            None,
                        ),
                        float(
                            "pipeline.actuation_feedback_delay_ms",
                            "设备移动的最小视觉反馈等待",
                            0.0,
                            100.0,
                            Some("ms"),
                        ),
                    ],
                ),
                section(
                    "paths",
                    "持久化路径",
                    vec![
                        string("paths.data_dir", "数据目录"),
                        string("paths.model_dir", "模型目录"),
                        string("paths.database", "数据库路径"),
                        string("paths.license", "许可证路径"),
                    ],
                ),
                section(
                    "capture",
                    "Jetson 采集",
                    vec![
                        string("capture.device", "V4L2 设备"),
                        select(
                            "capture.preference",
                            "采集策略",
                            &[
                                "auto_high_fps",
                                "auto_low_latency",
                                "auto_balanced",
                                "manual",
                            ],
                        ),
                        invariant_bool(
                            "capture.latest_only",
                            "仅保留最新帧",
                            "生产热路径要求为 true",
                        ),
                        invariant_int(
                            "capture.appsink_max_buffers",
                            "appsink 容量",
                            1.0,
                            "容量一是最新帧语义的一部分",
                        ),
                        integer(
                            "capture.width",
                            "采集宽度",
                            0.0,
                            u32::MAX as f64,
                            Some("px"),
                        ),
                        integer(
                            "capture.height",
                            "采集高度",
                            0.0,
                            u32::MAX as f64,
                            Some("px"),
                        ),
                        integer("capture.fps", "采集帧率", 0.0, u32::MAX as f64, Some("Hz")),
                        string("capture.pixel_format", "像素格式"),
                        integer(
                            "capture.roi_left",
                            "ROI 左偏移",
                            0.0,
                            u32::MAX as f64,
                            Some("px"),
                        ),
                        integer(
                            "capture.roi_top",
                            "ROI 上偏移",
                            0.0,
                            u32::MAX as f64,
                            Some("px"),
                        ),
                        integer(
                            "capture.roi_width",
                            "ROI 宽度",
                            0.0,
                            u32::MAX as f64,
                            Some("px"),
                        ),
                        integer(
                            "capture.roi_height",
                            "ROI 高度",
                            0.0,
                            u32::MAX as f64,
                            Some("px"),
                        ),
                    ],
                ),
                section(
                    "inference",
                    "GPU 推理",
                    vec![
                        boolean("inference.enabled", "启用推理"),
                        select("inference.backend", "推理后端", &["deepstream_nvinfer"]),
                        invariant_bool(
                            "inference.require_gpu",
                            "要求 GPU",
                            "Jetson 生产路径要求为 true",
                        ),
                        invariant_bool(
                            "inference.allow_cpu_fallback",
                            "允许 CPU 回退",
                            "Jetson 生产路径要求为 false",
                        ),
                        float(
                            "inference.inference_input_deadline_ms",
                            "推理输入时限",
                            f64::MIN_POSITIVE,
                            u32::MAX as f64,
                            Some("ms"),
                        ),
                        string("inference.deepstream_parser_library", "DeepStream 解析器库"),
                        integer(
                            "inference.deepstream_startup_timeout_ms",
                            "启动超时",
                            1.0,
                            u64::MAX as f64,
                            Some("ms"),
                        ),
                        integer(
                            "inference.deepstream_shutdown_timeout_ms",
                            "关闭超时",
                            1.0,
                            u64::MAX as f64,
                            Some("ms"),
                        ),
                    ],
                ),
                section(
                    "hardware",
                    "kmNet 输出",
                    vec![
                        boolean("hardware.auto_connect", "运行时自动连接"),
                        select("hardware.backend", "设备后端", &["native_udp"]),
                        string("hardware.host", "设备地址"),
                        integer("hardware.port", "设备端口", 1.0, 65_535.0, None),
                        string("hardware.uuid", "设备 UUID"),
                        integer("hardware.monitor_port", "监控端口", 1_024.0, 49_151.0, None),
                        integer(
                            "hardware.connect_timeout_ms",
                            "连接超时",
                            1.0,
                            u64::MAX as f64,
                            Some("ms"),
                        ),
                        integer(
                            "hardware.send_timeout_ms",
                            "发送超时",
                            1.0,
                            u64::MAX as f64,
                            Some("ms"),
                        ),
                        integer(
                            "hardware.monitor_timeout_ms",
                            "监控超时",
                            1.0,
                            u64::MAX as f64,
                            Some("ms"),
                        ),
                        integer(
                            "hardware.trigger_poll_interval_ms",
                            "触发轮询间隔",
                            1.0,
                            50.0,
                            Some("ms"),
                        ),
                    ],
                ),
            ],
        }
    }
}

fn section(
    id: &'static str,
    label: &'static str,
    fields: Vec<ConfigFieldSchema>,
) -> ConfigSectionSchema {
    ConfigSectionSchema { id, label, fields }
}

fn runtime_section(
    id: &'static str,
    label: &'static str,
    mut fields: Vec<ConfigFieldSchema>,
) -> ConfigSectionSchema {
    for field in &mut fields {
        field.restart_required = false;
    }
    ConfigSectionSchema { id, label, fields }
}

fn field(path: &'static str, label: &'static str, field_type: &'static str) -> ConfigFieldSchema {
    ConfigFieldSchema {
        path,
        label,
        field_type,
        options: None,
        min: None,
        max: None,
        unit: None,
        description: None,
        restart_required: true,
    }
}

fn string(path: &'static str, label: &'static str) -> ConfigFieldSchema {
    field(path, label, "string")
}

fn boolean(path: &'static str, label: &'static str) -> ConfigFieldSchema {
    field(path, label, "bool")
}

fn hot_boolean(path: &'static str, label: &'static str) -> ConfigFieldSchema {
    ConfigFieldSchema {
        restart_required: false,
        ..boolean(path, label)
    }
}

fn select(
    path: &'static str,
    label: &'static str,
    options: &'static [&'static str],
) -> ConfigFieldSchema {
    ConfigFieldSchema {
        options: Some(options),
        ..field(path, label, "select")
    }
}

fn hot_select(
    path: &'static str,
    label: &'static str,
    options: &'static [&'static str],
) -> ConfigFieldSchema {
    ConfigFieldSchema {
        restart_required: false,
        ..select(path, label, options)
    }
}

fn integer(
    path: &'static str,
    label: &'static str,
    min: f64,
    max: f64,
    unit: Option<&'static str>,
) -> ConfigFieldSchema {
    ConfigFieldSchema {
        min: Some(min),
        max: Some(max),
        unit,
        ..field(path, label, "int")
    }
}

fn float(
    path: &'static str,
    label: &'static str,
    min: f64,
    max: f64,
    unit: Option<&'static str>,
) -> ConfigFieldSchema {
    ConfigFieldSchema {
        min: Some(min),
        max: Some(max),
        unit,
        ..field(path, label, "float")
    }
}

fn invariant_bool(
    path: &'static str,
    label: &'static str,
    description: &'static str,
) -> ConfigFieldSchema {
    ConfigFieldSchema {
        description: Some(description),
        ..boolean(path, label)
    }
}

fn invariant_int(
    path: &'static str,
    label: &'static str,
    value: f64,
    description: &'static str,
) -> ConfigFieldSchema {
    ConfigFieldSchema {
        description: Some(description),
        ..integer(path, label, value, value, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_exposes_only_persisted_rust_config_paths() {
        let schema = ConfigSchemaResponse::new(&AppConfig::default());
        let value = serde_json::to_value(schema).unwrap();

        assert_eq!(value["version"], 12);
        assert_eq!(value["algorithm"]["id"], "continuous_atan_predictive_v1");
        assert_eq!(value["algorithm"]["response"]["atan_scale_counts"], 256.0);
        assert_eq!(
            value["algorithm"]["response"]["static_acquisition_boost_fraction"],
            0.35
        );
        assert_eq!(
            value["algorithm"]["response"]["motion_boost_fraction"],
            0.65
        );
        assert_eq!(value["algorithm"]["prediction"]["aim_history_points"], 4);
        assert_eq!(value["algorithm"]["prediction"]["velocity_segments"], 3);
        assert_eq!(value["values"]["server"]["port"], 5174);
        assert_eq!(value["values"]["pipeline"]["arrival_radius_counts"], 3.0);
        assert_eq!(
            value["values"]["pipeline"]["actuation_feedback_delay_ms"],
            4.0
        );
        assert_eq!(value["values"]["pipeline"]["prediction_lead_ms"], 16.0);
        assert_eq!(value["values"]["pipeline"]["p_response_boost"], 0.5);
        assert_eq!(value["values"]["pipeline"]["max_counts_per_update"], 127.0);
        assert_eq!(value["values"]["pipeline"]["prediction_cap_px"], 10.0);
        assert_eq!(value["values"]["inference"], Value::Null);
        assert!(value["sections"].as_array().unwrap().iter().any(|section| {
            section["id"] == "inference"
                && section["fields"].as_array().unwrap().iter().any(|field| {
                    field["path"] == "inference.backend"
                        && field["options"] == serde_json::json!(["deepstream_nvinfer"])
                })
        }));
        assert!(value["sections"].as_array().unwrap().iter().any(|section| {
            section["id"] == "control.recoil"
                && section["fields"].as_array().unwrap().iter().any(|field| {
                    field["path"] == "control.recoil.interval_ms"
                        && field["min"] == 1.0
                        && field["max"] == 5_000.0
                        && field["restart_required"] == false
                })
        }));
        let output_gate = value["sections"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|section| section["fields"].as_array().unwrap())
            .find(|field| field["path"] == "control.output_enabled")
            .unwrap();
        assert_eq!(output_gate["restart_required"], false);
        let response_scale = value["sections"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|section| section["fields"].as_array().unwrap())
            .find(|field| field["path"] == "pipeline.p_response_scale")
            .unwrap();
        assert_eq!(response_scale["restart_required"], false);
        let hot_control_paths = [
            "control.output_enabled",
            "control.trigger_mode",
            "pipeline.freshness_threshold_ms",
            "pipeline.projection_fov_x_deg",
            "pipeline.projection_counts_per_360",
            "pipeline.p_response_scale",
            "pipeline.p_response_boost",
            "pipeline.p_response_curve_shape",
            "pipeline.max_counts_per_update",
            "pipeline.prediction_enabled",
            "pipeline.velocity_history_reset_gap_ms",
            "pipeline.velocity_spread_base_px_ms",
            "pipeline.velocity_spread_relative",
            "pipeline.prediction_lead_ms",
            "pipeline.prediction_cap_px",
            "pipeline.arrival_radius_counts",
            "pipeline.residual_cap",
            "pipeline.actuation_feedback_delay_ms",
            "pipeline.target_fov_radius_px",
            "pipeline.target_min_confidence",
            "pipeline.target_track_max_age",
            "pipeline.target_track_max_lost_age_ms",
            "pipeline.tracker_max_match_distance",
            "pipeline.tracker_position_cost_weight",
            "pipeline.tracker_iou_cost_weight",
            "pipeline.tracker_scale_cost_weight",
            "pipeline.tracker_max_size_ratio",
            "pipeline.tracker_max_association_dt_ms",
            "pipeline.tracker_kalman_acceleration_noise",
            "pipeline.tracker_kalman_measurement_noise_x",
            "pipeline.tracker_kalman_measurement_noise_y",
            "pipeline.tracker_kalman_max_predict_dt_ms",
            "pipeline.tracker_kalman_max_predict_missing_ms",
            "pipeline.tracker_kalman_max_predict_steps",
            "pipeline.tracker_kalman_nis_threshold",
            "pipeline.tracker_kalman_nis_hard_reject",
            "pipeline.target_class_priority",
            "pipeline.target_class_filter",
            "pipeline.target_selection_class_ratio",
            "pipeline.target_switch_min_preference_advantage",
            "pipeline.target_switch_min_continuity_score",
            "pipeline.target_switch_delay_ms",
            "pipeline.target_aim_y_ratio",
            "pipeline.target_class_aim_y_ratios",
            "pipeline.candidate_max_aspect_ratio",
        ];
        for path in hot_control_paths {
            let field = value["sections"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|section| section["fields"].as_array().unwrap())
                .find(|field| field["path"] == path)
                .unwrap_or_else(|| panic!("missing config schema field {path}"));
            assert_eq!(field["restart_required"], false, "{path} must be hot");
        }
        assert!(
            value["sections"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|section| section["fields"].as_array().unwrap())
                .all(|field| field["path"] != "pipeline.atan_scale_counts")
        );
        assert!(value["sections"].as_array().unwrap().iter().any(|section| {
            section["id"] == "crosshair"
                && section["fields"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|field| field["path"] == "crosshair.use_for_control")
        }));
    }
}
