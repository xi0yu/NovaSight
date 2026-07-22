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
                            "pipeline.velocity_smoothing_frames",
                            "速度 EMA 平滑帧数",
                            0.000_001,
                            1_000.0,
                            Some("frame"),
                        ),
                        float(
                            "pipeline.velocity_history_reset_gap_ms",
                            "速度历史重置间隔",
                            0.000_001,
                            10_000.0,
                            Some("ms"),
                        ),
                        float(
                            "pipeline.velocity_spread_base_px_ms",
                            "速度离散基础尺度",
                            0.000_001,
                            10_000.0,
                            Some("px/ms"),
                        ),
                        float(
                            "pipeline.velocity_spread_relative",
                            "速度离散相对尺度",
                            0.0,
                            1_000.0,
                            None,
                        ),
                        float(
                            "pipeline.velocity_change_base_px_ms",
                            "速度变化基础尺度",
                            0.000_001,
                            10_000.0,
                            Some("px/ms"),
                        ),
                        float(
                            "pipeline.velocity_change_relative",
                            "速度变化相对尺度",
                            0.0,
                            1_000.0,
                            None,
                        ),
                        float(
                            "pipeline.prediction_lead_frames",
                            "预测提前帧数",
                            0.0,
                            10.0,
                            Some("frame"),
                        ),
                        float(
                            "pipeline.prediction_far_absolute_cap_px",
                            "远目标预测绝对上限",
                            0.0,
                            100_000.0,
                            Some("px"),
                        ),
                        float(
                            "pipeline.prediction_far_base_cap_px",
                            "远目标预测基础上限",
                            0.0,
                            100_000.0,
                            Some("px"),
                        ),
                        float(
                            "pipeline.prediction_far_relative_cap",
                            "远目标预测相对上限",
                            0.0,
                            100_000.0,
                            None,
                        ),
                        float(
                            "pipeline.prediction_near_absolute_cap_px",
                            "近目标预测绝对上限",
                            0.0,
                            100_000.0,
                            Some("px"),
                        ),
                        float(
                            "pipeline.prediction_near_base_cap_px",
                            "近目标预测基础上限",
                            0.0,
                            100_000.0,
                            Some("px"),
                        ),
                        float(
                            "pipeline.prediction_near_relative_cap",
                            "近目标预测相对上限",
                            0.0,
                            100_000.0,
                            None,
                        ),
                        float(
                            "pipeline.residual_cap",
                            "量化残差上限",
                            0.0,
                            1.0,
                            Some("count"),
                        ),
                        float(
                            "pipeline.target_debounce_distance_px",
                            "目标锁定防抖距离",
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
                        integer(
                            "pipeline.max_command_age_ms",
                            "设备命令最大年龄",
                            1.0,
                            1_000.0,
                            Some("ms"),
                        ),
                        integer(
                            "pipeline.output_interval_ms",
                            "输出调度间隔",
                            1.0,
                            10.0,
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
                        select("capture.backend", "采集后端", &["deepstream_nvinfer"]),
                        select("capture.memory", "内存路径", &["nvmm"]),
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
                        select("capture.queue_leaky", "队列丢弃策略", &["downstream"]),
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
                        select("inference.device", "计算设备", &["cuda"]),
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
                            "inference.confidence_threshold",
                            "置信度阈值",
                            0.0,
                            1.0,
                            None,
                        ),
                        float("inference.nms_threshold", "NMS 阈值", 0.0, 1.0, None),
                        float(
                            "inference.inference_input_deadline_ms",
                            "推理输入时限",
                            f64::MIN_POSITIVE,
                            u32::MAX as f64,
                            Some("ms"),
                        ),
                        string("inference.deepstream_parser_library", "DeepStream 解析器库"),
                        integer(
                            "inference.deepstream_io_mode",
                            "DeepStream I/O 模式",
                            0.0,
                            i32::MAX as f64,
                            None,
                        ),
                        integer(
                            "inference.deepstream_batched_push_timeout_us",
                            "批推送超时",
                            0.0,
                            i64::MAX as f64,
                            Some("us"),
                        ),
                        integer(
                            "inference.deepstream_component_id",
                            "推理组件 ID",
                            0.0,
                            i32::MAX as f64,
                            None,
                        ),
                        integer(
                            "inference.deepstream_source_id",
                            "输入源 ID",
                            0.0,
                            u32::MAX as f64,
                            None,
                        ),
                        string("inference.deepstream_probe_element", "探针元素"),
                        string("inference.deepstream_probe_pad", "探针 Pad"),
                        string("inference.deepstream_nvinfer_config", "nvinfer 配置"),
                        integer(
                            "inference.model_width",
                            "模型宽度",
                            1.0,
                            u32::MAX as f64,
                            Some("px"),
                        ),
                        integer(
                            "inference.model_height",
                            "模型高度",
                            1.0,
                            u32::MAX as f64,
                            Some("px"),
                        ),
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
                        select("inference.input_source", "推理输入源", &["source.default"]),
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

        assert_eq!(value["version"], 1);
        assert_eq!(value["values"]["server"]["port"], 5174);
        assert_eq!(value["values"]["pipeline"]["max_command_age_ms"], 55);
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
        assert!(value["sections"].as_array().unwrap().iter().all(|section| {
            section["fields"].as_array().unwrap().iter().all(|field| {
                field["restart_required"] == true
                    && !field["path"].as_str().unwrap().starts_with("crosshair.")
            })
        }));
    }
}
