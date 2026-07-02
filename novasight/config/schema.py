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
                "id": "control",
                "label": "控制",
                "fields": [
                    {"path": "control.max_abs_dx", "label": "X 限幅", "type": "int", "min": 0, "restart_required": False},
                    {"path": "control.max_abs_dy", "label": "Y 限幅", "type": "int", "min": 0, "restart_required": False},
                    {"path": "control.min_confidence", "label": "最低置信度", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.strategy", "label": "控制策略", "type": "select", "options": ["pid", "predictive"], "restart_required": False},
                    {"path": "control.command_interval_ms", "label": "指令合并间隔 ms", "type": "float", "min": 0, "restart_required": False},
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
