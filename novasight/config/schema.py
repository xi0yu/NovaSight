from __future__ import annotations

from dataclasses import asdict
from typing import Any

from novasight.config.params import param_schema_for
from novasight.config.runtime import RuntimeConfig


def runtime_config_schema(config: RuntimeConfig | None = None) -> dict[str, Any]:
    cfg = config or RuntimeConfig()
    schema = {
        "version": 1,
        "values": {},
        "sections": [
            {
                "id": "web",
                "label": "Web 服务",
                "fields": [
                    {"path": "web.host", "label": "监听地址", "type": "string", "restart_required": True},
                    {"path": "web.port", "label": "监听端口", "type": "int", "min": 1, "max": 65535, "restart_required": True},
                ],
            },
            {
                "id": "source",
                "label": "输入源",
                "fields": [
                    {
                        "path": "source.default",
                        "label": "默认输入源",
                        "type": "select",
                        "options": ["null", "capture", "image"],
                        "restart_required": False,
                    },
                    {"path": "source.target_fps", "label": "目标帧率", "type": "int", "min": 1, "max": 240, "restart_required": False},
                    {"path": "source.image_path", "label": "图片路径", "type": "string", "restart_required": False},
                    {
                        "path": "source.image_fps",
                        "label": "图片循环帧率",
                        "type": "select",
                        "options": ["1", "5", "15", "30", "60"],
                        "restart_required": False,
                    },
                ],
            },
            {
                "id": "consumers",
                "label": "消费者",
                "fields": [
                    {"path": "consumers.preview", "label": "浏览器预览", "type": "bool", "restart_required": False},
                    {"path": "consumers.inference", "label": "模型推理", "type": "bool", "restart_required": False},
                    {"path": "consumers.recording", "label": "录制回放", "type": "bool", "restart_required": False},
                    {
                        "path": "consumers.recording_format",
                        "label": "录制格式",
                        "type": "select",
                        "options": ["csv", "parquet"],
                        "restart_required": True,
                    },
                    {
                        "path": "consumers.recording_path",
                        "label": "录制文件路径",
                        "type": "string",
                        "restart_required": True,
                    },
                ],
            },
            {
                "id": "capture",
                "label": "采集",
                "fields": [
                    {"path": "capture.device", "label": "设备路径", "type": "string", "restart_required": True},
                    {
                        "path": "capture.backend",
                        "label": "采集后端",
                        "type": "select",
                        "options": ["gst_cpu_latest", "nvmm_latest"],
                        "restart_required": True,
                    },
                    {
                        "path": "capture.preference",
                        "label": "选择策略",
                        "type": "select",
                        "options": ["auto_high_fps", "auto_low_latency", "auto_balanced", "manual"],
                        "restart_required": True,
                    },
                    {
                        "path": "capture.memory",
                        "label": "采集内存路径",
                        "type": "select",
                        "options": ["system", "nvmm"],
                        "restart_required": True,
                    },
                    {"path": "capture.latest_only", "label": "仅最新帧", "type": "bool", "restart_required": True},
                    {"path": "capture.appsink_max_buffers", "label": "appsink 缓冲", "type": "int", "min": 1, "max": 1, "restart_required": True},
                    {
                        "path": "capture.queue_leaky",
                        "label": "队列丢弃策略",
                        "type": "select",
                        "options": ["downstream"],
                        "restart_required": True,
                    },
                    {"path": "capture.pixel_format", "label": "像素格式", "type": "string", "restart_required": True},
                    {"path": "capture.width", "label": "宽度", "type": "int", "min": 0, "restart_required": True},
                    {"path": "capture.height", "label": "高度", "type": "int", "min": 0, "restart_required": True},
                    {"path": "capture.fps", "label": "帧率", "type": "int", "min": 0, "restart_required": True},
                ],
            },
            {
                "id": "preprocess",
                "label": "预处理",
                "fields": [
                    {
                        "path": "preprocess.backend",
                        "label": "预处理后端",
                        "type": "select",
                        "options": ["cpu", "cuda"],
                        "restart_required": True,
                    },
                    {
                        "path": "preprocess.input_format",
                        "label": "输入格式",
                        "type": "select",
                        "options": ["auto"],
                        "restart_required": True,
                    },
                    {
                        "path": "preprocess.output_dtype",
                        "label": "输出精度",
                        "type": "select",
                        "options": ["fp16", "fp32"],
                        "restart_required": True,
                    },
                    {"path": "preprocess.normalize", "label": "归一化", "type": "bool", "restart_required": True},
                    {"path": "preprocess.use_pinned_memory", "label": "Pinned Host Memory", "type": "bool", "restart_required": True},
                    {"path": "preprocess.h2d_async", "label": "异步 H2D", "type": "bool", "restart_required": True},
                ],
            },
            {
                "id": "calibration",
                "label": "标定 Profile",
                "fields": [
                    {"path": "calibration.profile_id", "label": "Profile ID", "type": "string", "restart_required": False},
                    {"path": "calibration.profile_version", "label": "Profile 版本", "type": "int", "min": 1, "restart_required": False},
                    {
                        "path": "calibration.fov_semantics",
                        "label": "FOV 语义",
                        "type": "select",
                        "options": ["horizontal"],
                        "restart_required": False,
                    },
                    {"path": "calibration.fov_x_deg", "label": "水平 FOVX", "type": "float", "min": 30, "max": 179, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "calibration.counts_per_360_x", "label": "水平每圈 counts", "type": "float", "min": 1, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "calibration.counts_per_360_y", "label": "垂直每圈 counts", "type": "float", "min": 1, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "calibration.invert_y", "label": "反转 Y 轴", "type": "bool", "restart_required": False},
                    {"path": "calibration.game_sensitivity_fingerprint", "label": "灵敏度指纹", "type": "string", "restart_required": False},
                    {
                        "path": "calibration.projection_profile",
                        "label": "投影 Profile",
                        "type": "select",
                        "options": ["fixed_horizontal_fov"],
                        "restart_required": False,
                    },
                ],
            },
            {
                "id": "limits",
                "label": "预览输出",
                "fields": [
                    {
                        "path": "limits.stream_fps",
                        "label": "预览帧率",
                        "type": "select",
                        "options": ["15", "30", "60"],
                        "restart_required": False,
                    },
                ],
            },
            {
                "id": "runtime",
                "label": "运行时",
                "fields": [
                    {
                        "path": "runtime.freshness_threshold_ms",
                        "label": "批次新鲜度(ms)",
                        "type": "float",
                        "min": 0,
                        "max": 1000,
                        "restart_required": False,
                    },
                    {"path": "runtime.drop_stale_batches", "label": "丢弃过期批次", "type": "bool", "restart_required": False},
                    {"path": "runtime.consume_latest_only", "label": "仅消费最新帧", "type": "bool", "restart_required": True},
                ],
            },
            {
                "id": "roi",
                "label": "ROI",
                "fields": [
                    {
                        "path": "roi.size",
                        "label": "中心 ROI",
                        "type": "select",
                        "options": ["640", "480", "320", "256"],
                        "restart_required": False,
                    },
                    {"path": "roi.offset_x", "label": "ROI 水平偏移", "type": "int", "min": -4096, "max": 4096, "restart_required": False},
                    {"path": "roi.offset_y", "label": "ROI 垂直偏移", "type": "int", "min": -4096, "max": 4096, "restart_required": False},
                ],
            },
            {
                "id": "inference",
                "label": "推理",
                "fields": [
                    {"path": "inference.enabled", "label": "启用推理", "type": "bool", "restart_required": False},
                    {
                        "path": "inference.backend",
                        "label": "推理后端",
                        "type": "select",
                        "options": ["tensorrt", "nvmm_latest"],
                        "restart_required": True,
                    },
                    {
                        "path": "inference.device",
                        "label": "推理设备",
                        "type": "select",
                        "options": ["cuda"],
                        "restart_required": True,
                    },
                    {"path": "inference.require_gpu", "label": "必须使用 GPU", "type": "bool", "restart_required": True},
                    {"path": "inference.allow_cpu_fallback", "label": "允许 CPU 回退", "type": "bool", "restart_required": True},
                    {
                        "path": "inference.inference_input_deadline_ms",
                        "label": "推理输入新鲜度(ms)",
                        "type": "float",
                        "min": 0,
                        "max": 1000,
                        "restart_required": False,
                    },
                    {"path": "inference.confidence_threshold", "label": "置信度阈值", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "inference.nms_threshold", "label": "NMS 阈值", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "inference.detection_class_profile", "label": "检测类别配置", "type": "string", "restart_required": False},
                    {"path": "inference.detection_class_filter", "label": "检测类别过滤", "type": "string", "restart_required": False},
                    {"path": "inference.detection_class_priority", "label": "类别优先级", "type": "string", "restart_required": False},
                    {
                        "path": "inference.input_source",
                        "label": "推理输入源",
                        "type": "select",
                        "options": ["source.default"],
                        "restart_required": False,
                    },
                ],
            },
            {
                "id": "control",
                "label": "控制",
                "fields": [
                    {"path": "control.aim.y_ratio", "label": "瞄点纵向比例", "type": "float", "min": 0, "max": 1, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.configured_actuation_delay_s", "label": "估计执行延迟 s", "type": "float", "min": 0, "max": 0.1, "step": 0.001, "precision": 3, "restart_required": False},
                    {"path": "control.prediction_strength", "label": "预测强度", "type": "float", "min": 0, "max": 1.5, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.prediction_x_enabled", "label": "预测 X", "type": "bool", "restart_required": False},
                    {"path": "control.prediction_y_enabled", "label": "预测 Y", "type": "bool", "restart_required": False},
                    {"path": "control.kp_x", "label": "Kp X", "type": "float", "min": 0, "max": 2, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.kp_y", "label": "Kp Y", "type": "float", "min": 0, "max": 2, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.kd_x", "label": "Kd X", "type": "float", "min": 0, "max": 1, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.kd_y", "label": "Kd Y", "type": "float", "min": 0, "max": 1, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.d_ema_alpha", "label": "D 项 EMA", "type": "float", "min": 0.01, "max": 1, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.deadzone_px_x", "label": "X 像素死区", "type": "float", "min": 0, "max": 10, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.deadzone_px_y", "label": "Y 像素死区", "type": "float", "min": 0, "max": 10, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.max_output_rad_x", "label": "X 单次最大角度", "type": "float", "min": 0.0001, "step": 0.0001, "precision": 4, "restart_required": False},
                    {"path": "control.max_output_rad_y", "label": "Y 单次最大角度", "type": "float", "min": 0.0001, "step": 0.0001, "precision": 4, "restart_required": False},
                    {"path": "control.max_output_rate_rad_s_x", "label": "X 输出变化率 rad/s", "type": "float", "min": 0.0001, "step": 0.0001, "precision": 4, "restart_required": False},
                    {"path": "control.max_output_rate_rad_s_y", "label": "Y 输出变化率 rad/s", "type": "float", "min": 0.0001, "step": 0.0001, "precision": 4, "restart_required": False},
                    {"path": "control.scheduler_step_counts_x", "label": "Scheduler X 单步 counts", "type": "int", "min": 1, "max": 20, "step": 1, "restart_required": False},
                    {"path": "control.scheduler_step_counts_y", "label": "Scheduler Y 单步 counts", "type": "int", "min": 1, "max": 20, "step": 1, "restart_required": False},
                    {"path": "control.scheduler_interval_ms", "label": "Scheduler 间隔 ms", "type": "float", "min": 1, "max": 10, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.target_fov_radius_px", "label": "目标选择半径 px", "type": "float", "min": 1, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "control.min_confidence", "label": "最低置信度", "type": "float", "min": 0.1, "max": 0.99, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.target_switch_delay_ms", "label": "目标切换延迟 ms", "type": "float", "min": 0, "max": 500, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "control.lost_target_timeout_ms", "label": "目标丢失超时 ms", "type": "float", "min": 0, "max": 200, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "control.trigger_mode", "label": "触发方式", "type": "select", "options": ["hardware", "always"], "restart_required": False},
                    {"path": "control.output_mode", "label": "输出模式", "type": "select", "options": ["kmnet"], "restart_required": False},
                ],
            },
            {
                "id": "executor",
                "label": "执行器",
                "fields": [
                    {"path": "executor.default", "label": "默认执行器", "type": "select", "options": ["kmnet"], "restart_required": False},
                ],
            },
            {
                "id": "hardware",
                "label": "硬件盒子",
                "fields": [
                    {"path": "hardware.kind", "label": "硬件类型", "type": "select", "options": ["kmnet"], "restart_required": True},
                    {"path": "hardware.auto_connect", "label": "服务启动自动连接", "type": "bool", "restart_required": False},
                    {"path": "hardware.host", "label": "盒子地址", "type": "string", "restart_required": True},
                    {"path": "hardware.port", "label": "盒子端口", "type": "int", "min": 0, "max": 65535, "restart_required": True},
                    {"path": "hardware.uuid", "label": "kmNet UUID", "type": "string", "restart_required": True},
                    {"path": "hardware.monitor_port", "label": "kmNet 监听端口", "type": "int", "min": 0, "max": 65535, "restart_required": True},
                    {"path": "hardware.serial_port", "label": "串口", "type": "string", "restart_required": True},
                    {"path": "hardware.heartbeat_timeout_ms", "label": "心跳超时 ms", "type": "float", "min": 1, "restart_required": False},
                ],
            },
            {
                "id": "logging",
                "label": "日志",
                "fields": [
                    {"path": "logging.level", "label": "日志级别", "type": "select", "options": ["DEBUG", "INFO", "WARNING", "ERROR"], "restart_required": False},
                    {"path": "logging.dir", "label": "日志目录", "type": "string", "restart_required": False},
                ],
            },
        ],
    }
    for section in schema["sections"]:
        for field in section["fields"]:
            spec = param_schema_for(str(field.get("path", "")))
            if not spec:
                continue
            field["label"] = spec["label"]
            field["default"] = spec["default"]
            field["min"] = spec["minimum"]
            field["max"] = spec["maximum"]
            field["recommended_min"] = spec["recommended_min"]
            field["recommended_max"] = spec["recommended_max"]
            field["step"] = spec["step"]
            field["unit"] = spec["unit"]
            field["description"] = spec["description"]
    schema["values"] = _visible_values(asdict(cfg), schema["sections"])
    return schema


def _visible_values(values: dict[str, Any], sections: list[dict[str, Any]]) -> dict[str, Any]:
    visible: dict[str, Any] = {}
    missing = object()
    for section in sections:
        for field in section.get("fields", []):
            path = str(field.get("path", ""))
            parts = path.split(".")
            if len(parts) < 2:
                continue
            source: Any = values
            for part in parts:
                if not isinstance(source, dict) or part not in source:
                    source = missing
                    break
                source = source[part]
            if source is missing:
                continue
            target = visible
            for part in parts[:-1]:
                child = target.setdefault(part, {})
                if not isinstance(child, dict):
                    break
                target = child
            else:
                target[parts[-1]] = source
    return visible
