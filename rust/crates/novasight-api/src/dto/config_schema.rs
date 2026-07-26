use novasight_runtime::AppConfig;
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ConfigSchemaResponse {
    version: u32,
    values: Value,
    sections: Vec<ConfigSectionSchema>,
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
                        select(
                            "control.trigger_mode",
                            "输出触发方式",
                            &["always", "hardware"],
                        ),
                    ],
                ),
                section(
                    "control.recoil",
                    "独立压枪",
                    vec![
                        boolean("control.recoil.enabled", "启用独立 Y 轴压枪"),
                        float(
                            "control.recoil.base_rate_counts_s",
                            "基础压枪速率",
                            0.0,
                            20_000.0,
                            Some("counts/s"),
                        ),
                        float(
                            "control.recoil.max_rate_counts_s",
                            "最大压枪速率",
                            0.0,
                            20_000.0,
                            Some("counts/s"),
                        ),
                        float(
                            "control.recoil.startup_ms",
                            "启动斜坡",
                            0.0,
                            1_000.0,
                            Some("ms"),
                        ),
                        float(
                            "control.recoil.positive_deadzone_norm",
                            "追加起点",
                            0.0,
                            1.0,
                            None,
                        ),
                        float(
                            "control.recoil.negative_deadzone_norm",
                            "刹车起点",
                            0.0,
                            1.0,
                            None,
                        ),
                        float(
                            "control.recoil.full_brake_error_norm",
                            "完全刹车误差",
                            0.0,
                            1.0,
                            None,
                        ),
                        float(
                            "control.recoil.fast_add_gain_counts_s",
                            "目标误差追加强度",
                            0.0,
                            20_000.0,
                            Some("counts/s"),
                        ),
                        float(
                            "control.recoil.max_fast_add_ratio",
                            "追加速率上限比例",
                            0.0,
                            1.0,
                            None,
                        ),
                        float(
                            "control.recoil.stale_threshold_ms",
                            "目标观测有效期",
                            0.0,
                            5_000.0,
                            Some("ms"),
                        ),
                    ],
                ),
                section(
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
                            "pipeline.near_threshold_px",
                            "近目标阈值",
                            0.0,
                            10_000.0,
                            Some("px"),
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
                        boolean("pipeline.projection_invert_y", "反转垂直输出"),
                        float(
                            "pipeline.atan_scale_counts",
                            "Atan 响应尺度",
                            0.000_001,
                            1_000_000.0,
                            Some("count"),
                        ),
                        float("pipeline.far_kp", "远目标反馈增益", 0.0, 100.0, None),
                        float(
                            "pipeline.far_max_counts_per_update",
                            "远目标单次计数上限",
                            1.0,
                            i16::MAX as f64,
                            Some("count"),
                        ),
                        float("pipeline.near_kp", "近目标反馈增益", 0.0, 100.0, None),
                        float(
                            "pipeline.near_max_counts_per_update",
                            "近目标单次计数上限",
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
                            "目标最大丢失帧数",
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
                        string("pipeline.target_class_priority", "目标类别优先级"),
                        string("pipeline.target_class_filter", "参与目标选择的类别"),
                        float(
                            "pipeline.target_selection_class_weight",
                            "目标类别评分权重",
                            0.0,
                            100.0,
                            None,
                        ),
                        float(
                            "pipeline.target_selection_distance_weight",
                            "目标距离评分权重",
                            0.0,
                            100.0,
                            None,
                        ),
                        float(
                            "pipeline.target_sticky_bias",
                            "已锁定目标粘滞偏置",
                            0.0,
                            0.9,
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
                        integer(
                            "pipeline.max_command_age_ms",
                            "设备命令最大年龄",
                            1.0,
                            1_000.0,
                            Some("ms"),
                        ),
                        integer(
                            "pipeline.output_interval_ms",
                            "空闲／后坐力调度间隔",
                            1.0,
                            10.0,
                            Some("ms"),
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
                        string("paths.python_executable", "Python 回退解释器"),
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
                        select(
                            "inference.backend",
                            "推理后端",
                            &["deepstream_nvinfer", "rust_tensor_rt"],
                        ),
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
                        select(
                            "hardware.backend",
                            "设备后端",
                            &["python_host", "native_udp"],
                        ),
                        string("hardware.host", "设备地址"),
                        integer("hardware.port", "设备端口", 1.0, 65_535.0, None),
                        string("hardware.uuid", "设备 UUID"),
                        integer("hardware.monitor_port", "监控端口", 1_024.0, 49_151.0, None),
                        string("hardware.helper_module", "Python 回退模块"),
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
                        integer(
                            "hardware.reconnect_cooldown_ms",
                            "重连冷却",
                            1.0,
                            u64::MAX as f64,
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

        assert_eq!(value["version"], 6);
        assert_eq!(value["values"]["server"]["port"], 5174);
        assert_eq!(value["values"]["pipeline"]["max_command_age_ms"], 55);
        assert_eq!(value["values"]["pipeline"]["arrival_radius_counts"], 3.0);
        assert_eq!(
            value["values"]["pipeline"]["actuation_feedback_delay_ms"],
            4.0
        );
        assert_eq!(value["values"]["inference"], Value::Null);
        assert!(value["sections"].as_array().unwrap().iter().any(|section| {
            section["id"] == "inference"
                && section["fields"].as_array().unwrap().iter().any(|field| {
                    field["path"] == "inference.backend"
                        && field["options"]
                            == serde_json::json!(["deepstream_nvinfer", "rust_tensor_rt"])
                })
        }));
        assert!(value["sections"].as_array().unwrap().iter().any(|section| {
            section["id"] == "pipeline"
                && section["fields"].as_array().unwrap().iter().any(|field| {
                    field["path"] == "pipeline.output_interval_ms"
                        && field["min"] == 1.0
                        && field["max"] == 10.0
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
