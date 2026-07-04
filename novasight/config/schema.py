from __future__ import annotations

from dataclasses import asdict
from typing import Any

from novasight.config.runtime import RuntimeConfig


def runtime_config_schema(config: RuntimeConfig | None = None) -> dict[str, Any]:
    cfg = config or RuntimeConfig()
    return {
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
                    {"path": "control.target_sticky_bias", "label": "锁定偏好", "type": "float", "min": 0, "max": 0.9, "restart_required": False},
                    {"path": "control.target_lost_grace_frames", "label": "丢失容忍帧", "type": "int", "min": 0, "restart_required": False},
                    {"path": "control.aim_ratio", "label": "瞄准高度", "type": "float", "min": 0, "max": 100, "restart_required": False},
                    {"path": "control.trigger_mode", "label": "触发方式", "type": "select", "options": ["hardware", "telemetry", "always"], "restart_required": False},
                    {"path": "control.trigger_bindings", "label": "本地触发绑定", "type": "string_list", "restart_required": False},
                    {"path": "control.strategy", "label": "控制策略", "type": "select", "options": ["straight", "pid", "proportional", "predictive"], "restart_required": False},
                    {"path": "control.pid_kp_x", "label": "PID Kp X轴", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.pid_kp_y", "label": "PID Kp Y轴", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.pid_ki", "label": "PID Ki", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.pid_kd", "label": "PID Kd", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.pid_integral_limit", "label": "PID 积分上限", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.pid_move_limit", "label": "PID 控制量上限", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.kp_x_move_max", "label": "Kp X 控制量上限", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.kp_y_move_max", "label": "Kp Y 控制量上限", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.prediction_factor", "label": "预测系数", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.command_interval_ms", "label": "指令合并间隔 ms", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.y_rate_window_ms", "label": "Y 轴压制窗口 ms", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.y_rate_max_counts", "label": "Y 轴窗口 counts 上限", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.move_kind", "label": "移动方式", "type": "select", "options": ["raw", "enc_raw", "auto", "enc_auto", "bezier", "enc_bezier"], "restart_required": False},
                    {"path": "control.move_ms", "label": "移动铺展 ms", "type": "int", "min": 0, "restart_required": False},
                    {"path": "control.trace_ms", "label": "kmNet trace ms", "type": "int", "min": 0, "restart_required": False},
                    {"path": "control.deadzone_counts", "label": "死区 counts", "type": "int", "min": 0, "restart_required": False},
                    {"path": "control.near_px", "label": "近距离 px", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.near_speed", "label": "近距离速度", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.far_speed", "label": "远距离速度", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.ema_alpha", "label": "EMA 平滑", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.counts_per_revolution_x", "label": "X 轴每圈 counts", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.counts_per_revolution_y", "label": "Y 轴每圈 counts", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.bezier_curvature", "label": "贝塞尔曲率", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.straight_fov_deg", "label": "Straight FOV", "type": "float", "min": 1, "max": 179, "restart_required": False},
                    {"path": "control.straight_c360", "label": "Straight 每圈 counts", "type": "float", "min": 100, "restart_required": False},
                    {"path": "control.straight_kp_x", "label": "Straight Kp X", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.straight_kp_y", "label": "Straight Kp Y", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.straight_first_frame_gain", "label": "Straight 首帧增益", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.straight_first_max_step", "label": "Straight 首帧上限", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.straight_max_step", "label": "Straight 精修上限", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.straight_in_deadzone", "label": "Straight 输入死区", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.straight_jump_threshold", "label": "Straight 跳变阈值", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.straight_pred_gain", "label": "Straight 预测强度", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.straight_pred_consistency_frames", "label": "Straight 动作判定帧", "type": "int", "min": 1, "restart_required": False},
                    {"path": "control.straight_gain_y", "label": "Straight Y 增益", "type": "float", "min": 0, "restart_required": False},
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
