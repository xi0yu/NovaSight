from __future__ import annotations

from dataclasses import asdict
from typing import Any

from novasight.config.params import param_schema_for
from novasight.config.runtime import RuntimeConfig


def runtime_config_schema(config: RuntimeConfig | None = None) -> dict[str, Any]:
    cfg = config or RuntimeConfig()
    schema = {
        "version": 1,
        "values": asdict(cfg),
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
                ],
            },
            {
                "id": "capture",
                "label": "采集",
                "fields": [
                    {"path": "capture.device", "label": "设备路径", "type": "string", "restart_required": True},
                    {
                        "path": "capture.preference",
                        "label": "选择策略",
                        "type": "select",
                        "options": ["auto_high_fps", "auto_low_latency", "auto_balanced", "manual"],
                        "restart_required": True,
                    },
                    {"path": "capture.pixel_format", "label": "像素格式", "type": "string", "restart_required": True},
                    {"path": "capture.width", "label": "宽度", "type": "int", "min": 0, "restart_required": True},
                    {"path": "capture.height", "label": "高度", "type": "int", "min": 0, "restart_required": True},
                    {"path": "capture.fps", "label": "帧率", "type": "int", "min": 0, "restart_required": True},
                ],
            },
            {
                "id": "limits",
                "label": "运行队列",
                "fields": [
                    {"path": "limits.max_frame_queue", "label": "队列容量", "type": "int", "min": 1, "max": 1, "restart_required": True},
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
                    {"path": "control.max_abs_dx", "label": "X 限幅", "type": "int", "min": 0, "restart_required": False},
                    {"path": "control.max_abs_dy", "label": "Y 限幅", "type": "int", "min": 0, "restart_required": False},
                    {"path": "control.min_confidence", "label": "最低置信度", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.fov_ratio", "label": "FOV 比例", "type": "float", "min": 0.01, "max": 1, "restart_required": False},
                    {"path": "control.target_lock_enabled", "label": "目标锁定", "type": "bool", "restart_required": False},
                    {"path": "control.target_lost_grace_frames", "label": "丢失容忍帧", "type": "int", "min": 0, "restart_required": False},
                    {"path": "control.aim_ratio", "label": "瞄准高度", "type": "float", "min": 0, "max": 100, "restart_required": False},
                    {"path": "control.strategy", "label": "鼠标算法", "type": "select", "options": ["pid", "isolated_mouse", "dynamic_pid"], "restart_required": False},
                    {"path": "control.trigger_mode", "label": "触发方式", "type": "select", "options": ["hardware", "telemetry", "always"], "restart_required": False},
                    {"path": "control.trigger_bindings", "label": "本地触发绑定", "type": "string_list", "restart_required": False},
                    {"path": "control.pid_kp_x", "label": "PID Kp X轴", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.pid_kp_y", "label": "PID Kp Y轴", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.pid_kd", "label": "PID Kd", "type": "float", "min": -1, "max": 1, "restart_required": False},
                    {"path": "control.kp_x_move_max", "label": "Kp X 控制量上限", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.kp_y_move_max", "label": "Kp Y 控制量上限", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.prediction_enabled", "label": "启用预测补偿", "type": "bool", "restart_required": False},
                    {"path": "control.prediction_factor", "label": "预测系数", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.derivative_enabled", "label": "启用 D 阻尼", "type": "bool", "restart_required": False},
                    {"path": "control.y_down_enabled", "label": "启用 Y 下压", "type": "bool", "restart_required": False},
                    {"path": "control.y_rate_window_ms", "label": "Y 轴下压间隔 ms", "type": "float", "min": 0, "max": 1000, "restart_required": False},
                    {"path": "control.y_rate_max_counts", "label": "Y 轴每次下压 counts", "type": "float", "min": 0, "max": 200, "restart_required": False},
                    {"path": "control.move_kind", "label": "移动方式", "type": "select", "options": ["raw", "enc_raw", "auto", "enc_auto", "bezier", "enc_bezier"], "restart_required": False},
                    {"path": "control.move_ms", "label": "移动铺展 ms", "type": "int", "min": 0, "restart_required": False},
                    {"path": "control.isolated_kp_x", "label": "隔离算法 Kp X", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.isolated_kp_y", "label": "隔离算法 Kp Y", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.isolated_max_x", "label": "隔离算法 X 上限", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.isolated_max_y", "label": "隔离算法 Y 上限", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.isolated_deadzone_px", "label": "隔离算法死区 px", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.isolated_aim_ratio", "label": "隔离算法瞄准高度", "type": "float", "min": 0, "max": 100, "restart_required": False},
                    {"path": "control.isolated_smoothing", "label": "隔离算法平滑", "type": "float", "min": 0, "max": 0.95, "restart_required": False},
                    {"path": "control.isolated_prediction", "label": "隔离算法预测", "type": "float", "min": 0, "max": 2, "restart_required": False},
                    {"path": "control.isolated_fov_deg", "label": "隔离算法 FOV", "type": "float", "min": 1, "max": 179, "restart_required": False},
                    {"path": "control.isolated_counts_per_revolution_x", "label": "隔离算法 X c360", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.isolated_counts_per_revolution_y", "label": "隔离算法 Y c360", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.dynamic_pid_kp_x", "label": "动态PID Kp X", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.dynamic_pid_kp_y", "label": "动态PID Kp Y", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.dynamic_pid_ki", "label": "动态PID Ki", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.dynamic_pid_kd", "label": "动态PID Kd", "type": "float", "min": -1, "max": 1, "restart_required": False},
                    {"path": "control.dynamic_pid_target_error_threshold", "label": "动态PID 达标误差阈值 rad", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.dynamic_pid_speed_multiplier", "label": "动态PID 速度倍率", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.dynamic_pid_min_coefficient", "label": "动态PID 最小系数", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.dynamic_pid_max_coefficient", "label": "动态PID 最大系数", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.dynamic_pid_transition_sharpness", "label": "动态PID 过渡锐度", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.dynamic_pid_transition_midpoint", "label": "动态PID 过渡中点", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.dynamic_pid_minimum_data_count", "label": "动态PID 最小数据量", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.dynamic_pid_error_change_tolerance", "label": "动态PID 误差变化容限 rad", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.dynamic_pid_smoothing_factor", "label": "动态PID 平滑因子", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.dynamic_pid_aim_ratio", "label": "动态PID 瞄准高度", "type": "float", "min": 0, "max": 100, "restart_required": False},
                    {"path": "control.dynamic_pid_fov_deg", "label": "动态PID FOV", "type": "float", "min": 1, "max": 179, "restart_required": False},
                    {"path": "control.dynamic_pid_counts_per_revolution_x", "label": "动态PID X 每圈 counts", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.dynamic_pid_counts_per_revolution_y", "label": "动态PID Y 每圈 counts", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.dynamic_pid_control_hz", "label": "动态PID 控制频率", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.dynamic_pid_ema_alpha", "label": "动态PID 目标滤波", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.output_mode", "label": "输出模式", "type": "select", "options": ["", "silent", "console", "dry_run", "kmnet"], "restart_required": False},
                ],
            },
            {
                "id": "executor",
                "label": "执行器",
                "fields": [
                    {"path": "executor.default", "label": "默认执行器", "type": "select", "options": ["silent", "console", "dry_run", "kmnet"], "restart_required": False},
                ],
            },
            {
                "id": "hardware",
                "label": "硬件盒子",
                "fields": [
                    {"path": "hardware.kind", "label": "硬件类型", "type": "select", "options": ["none", "kmnet", "makcu"], "restart_required": True},
                    {"path": "hardware.host", "label": "盒子地址", "type": "string", "restart_required": True},
                    {"path": "hardware.port", "label": "盒子端口", "type": "int", "min": 0, "max": 65535, "restart_required": True},
                    {"path": "hardware.uuid", "label": "kmNet UUID", "type": "string", "restart_required": True},
                    {"path": "hardware.monitor_port", "label": "kmNet 监听端口", "type": "int", "min": 0, "max": 65535, "restart_required": True},
                    {"path": "hardware.flip_dy", "label": "Y 轴反向", "type": "bool", "restart_required": False},
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
    return schema
