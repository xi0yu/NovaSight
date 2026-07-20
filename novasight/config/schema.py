from __future__ import annotations

from dataclasses import asdict
from typing import Any

from novasight.config.params import param_schema_for
from novasight.config.runtime import RuntimeConfig
from novasight.control.registry import (
    CALIBRATED_ANGULAR,
    DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2,
    UNIVERSAL_SATURATED,
    algorithm_definition,
    supported_algorithm_ids,
)


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
                    {
                        "path": "web.host",
                        "label": "监听地址",
                        "type": "string",
                        "restart_required": True,
                    },
                    {
                        "path": "web.port",
                        "label": "监听端口",
                        "type": "int",
                        "min": 1,
                        "max": 65535,
                        "restart_required": True,
                    },
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
                    {
                        "path": "source.image_path",
                        "label": "图片路径",
                        "type": "string",
                        "restart_required": False,
                    },
                ],
            },
            {
                "id": "consumers",
                "label": "消费者",
                "fields": [
                    {
                        "path": "consumers.preview",
                        "label": "浏览器预览",
                        "type": "bool",
                        "restart_required": False,
                    },
                    {
                        "path": "consumers.inference",
                        "label": "模型推理",
                        "type": "bool",
                        "restart_required": False,
                    },
                    {
                        "path": "consumers.recording",
                        "label": "录制回放",
                        "type": "bool",
                        "restart_required": False,
                    },
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
                    {
                        "path": "capture.device",
                        "label": "设备路径",
                        "type": "string",
                        "restart_required": True,
                    },
                    {
                        "path": "capture.backend",
                        "label": "采集后端",
                        "type": "select",
                        "options": ["deepstream_nvinfer"],
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
                        "options": ["nvmm"],
                        "restart_required": True,
                    },
                    {
                        "path": "capture.latest_only",
                        "label": "仅最新帧",
                        "type": "bool",
                        "restart_required": True,
                    },
                    {
                        "path": "capture.appsink_max_buffers",
                        "label": "appsink 缓冲",
                        "type": "int",
                        "min": 1,
                        "max": 1,
                        "restart_required": True,
                    },
                    {
                        "path": "capture.queue_leaky",
                        "label": "队列丢弃策略",
                        "type": "select",
                        "options": ["downstream"],
                        "restart_required": True,
                    },
                    {
                        "path": "capture.pixel_format",
                        "label": "像素格式",
                        "type": "string",
                        "restart_required": True,
                    },
                    {
                        "path": "capture.width",
                        "label": "宽度",
                        "type": "int",
                        "min": 0,
                        "restart_required": True,
                    },
                    {
                        "path": "capture.height",
                        "label": "高度",
                        "type": "int",
                        "min": 0,
                        "restart_required": True,
                    },
                    {
                        "path": "capture.fps",
                        "label": "帧率",
                        "type": "int",
                        "min": 0,
                        "restart_required": True,
                    },
                ],
            },
            {
                "id": "crosshair",
                "label": "准星控制基准",
                "fields": [
                    {
                        "path": "crosshair.enabled",
                        "label": "启用准星观测",
                        "type": "bool",
                        "restart_required": True,
                    },
                    {
                        "path": "crosshair.use_for_control",
                        "label": "使用已确认准星作为控制基准",
                        "type": "bool",
                        "restart_required": False,
                    },
                    {
                        "path": "crosshair.search_size",
                        "label": "中心搜索区尺寸",
                        "type": "int",
                        "min": 32,
                        "max": 640,
                        "step": 2,
                        "restart_required": True,
                    },
                    {
                        "path": "crosshair.sample_hz",
                        "label": "准星观测频率",
                        "type": "int",
                        "min": 1,
                        "max": 30,
                        "unit": "Hz",
                        "restart_required": True,
                    },
                    {
                        "path": "crosshair.sample_frames",
                        "label": "学习采样帧数",
                        "type": "int",
                        "min": 3,
                        "max": 15,
                        "restart_required": False,
                    },
                    {
                        "path": "crosshair.confirm_duration_ms",
                        "label": "稳定确认时间",
                        "type": "float",
                        "min": 0,
                        "max": 2000,
                        "unit": "ms",
                        "restart_required": False,
                    },
                    {
                        "path": "crosshair.min_similarity",
                        "label": "模板最低相似度",
                        "type": "float",
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
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
                        "options": ["cuda"],
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
                    {
                        "path": "preprocess.normalize",
                        "label": "归一化",
                        "type": "bool",
                        "restart_required": True,
                    },
                    {
                        "path": "preprocess.use_pinned_memory",
                        "label": "Pinned Host Memory",
                        "type": "bool",
                        "restart_required": True,
                    },
                    {
                        "path": "preprocess.h2d_async",
                        "label": "异步 H2D",
                        "type": "bool",
                        "restart_required": True,
                    },
                ],
            },
            {
                "id": "calibration",
                "label": "标定 Profile",
                "fields": [
                    {
                        "path": "calibration.profile_id",
                        "label": "Profile ID",
                        "type": "string",
                        "restart_required": False,
                    },
                    {
                        "path": "calibration.profile_version",
                        "label": "Profile 版本",
                        "type": "int",
                        "min": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "calibration.game_sensitivity_fingerprint",
                        "label": "灵敏度指纹",
                        "type": "string",
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
                        "options": ["15", "30"],
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
                    {
                        "path": "runtime.drop_stale_batches",
                        "label": "丢弃过期批次",
                        "type": "bool",
                        "restart_required": False,
                    },
                    {
                        "path": "runtime.consume_latest_only",
                        "label": "仅消费最新帧",
                        "type": "bool",
                        "restart_required": True,
                    },
                ],
            },
            {
                "id": "power_saving",
                "label": "主机离线省流",
                "fields": [
                    {
                        "path": "power_saving.host_presence_enabled",
                        "label": "启用主机心跳监管",
                        "type": "bool",
                        "restart_required": True,
                    },
                    {
                        "path": "power_saving.target_host_id",
                        "label": "目标主机 ID",
                        "type": "string",
                        "restart_required": True,
                    },
                    {
                        "path": "power_saving.heartbeat_timeout_s",
                        "label": "心跳超时",
                        "type": "float",
                        "min": 1,
                        "max": 120,
                        "unit": "s",
                        "restart_required": True,
                    },
                    {
                        "path": "power_saving.offline_grace_s",
                        "label": "离线宽限",
                        "type": "float",
                        "min": 0,
                        "max": 600,
                        "unit": "s",
                        "restart_required": True,
                    },
                    {
                        "path": "power_saving.auto_resume",
                        "label": "主机恢复后自动启动",
                        "type": "bool",
                        "restart_required": True,
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
                ],
            },
            {
                "id": "inference",
                "label": "推理",
                "fields": [
                    {
                        "path": "inference.enabled",
                        "label": "启用推理",
                        "type": "bool",
                        "restart_required": False,
                    },
                    {
                        "path": "inference.backend",
                        "label": "推理后端",
                        "type": "select",
                        "options": ["deepstream_nvinfer"],
                        "restart_required": True,
                    },
                    {
                        "path": "inference.device",
                        "label": "推理设备",
                        "type": "select",
                        "options": ["cuda"],
                        "restart_required": True,
                    },
                    {
                        "path": "inference.require_gpu",
                        "label": "必须使用 GPU",
                        "type": "bool",
                        "restart_required": True,
                    },
                    {
                        "path": "inference.inference_input_deadline_ms",
                        "label": "推理输入新鲜度(ms)",
                        "type": "float",
                        "min": 0,
                        "max": 1000,
                        "restart_required": False,
                    },
                    {
                        "path": "inference.confidence_threshold",
                        "label": "置信度阈值",
                        "type": "float",
                        "min": 0,
                        "max": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "inference.nms_threshold",
                        "label": "NMS 阈值",
                        "type": "float",
                        "min": 0,
                        "max": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "inference.deepstream_io_mode",
                        "label": "DeepStream V4L2 IO 模式",
                        "type": "int",
                        "min": 0,
                        "max": 5,
                        "restart_required": True,
                    },
                    {
                        "path": "inference.deepstream_batched_push_timeout_us",
                        "label": "DeepStream 批次等待 us",
                        "type": "int",
                        "min": 0,
                        "restart_required": True,
                    },
                    {
                        "path": "inference.deepstream_parser_library",
                        "label": "DeepStream C++ Parser",
                        "type": "string",
                        "restart_required": True,
                    },
                    {
                        "path": "inference.detection_class_profile",
                        "label": "检测类别配置",
                        "type": "string",
                        "restart_required": False,
                    },
                    {
                        "path": "inference.detection_class_filter",
                        "label": "检测类别过滤",
                        "type": "string",
                        "restart_required": False,
                    },
                    {
                        "path": "inference.detection_class_priority",
                        "label": "类别优先级",
                        "type": "string",
                        "restart_required": False,
                    },
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
                    {
                        "path": "control.active_algorithm",
                        "label": "控制算法",
                        "type": "select",
                        "options": [
                            "dual_phase_atan_robust_predictive_v2",
                            "universal_saturated",
                            "calibrated_angular",
                        ],
                        "restart_required": False,
                    },
                    *[
                        {
                            "path": f"control.aim.role_y_ratios.{role}",
                            "label": f"{label}瞄点纵向比例",
                            "type": "float",
                            "min": 0,
                            "max": 1,
                            "step": 0.01,
                            "precision": 2,
                            "restart_required": False,
                        }
                        for role, label in (("head", "头部"), ("body", "身体"), ("other", "其他"))
                    ],
                    {
                        "path": "control.configured_actuation_delay_s",
                        "label": "估计执行延迟 s",
                        "type": "float",
                        "min": 0,
                        "max": 0.1,
                        "step": 0.001,
                        "precision": 3,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.calibrated_angular.fov_x_deg",
                        "label": "精确角度控制：水平 FOVX",
                        "type": "float",
                        "min": 30,
                        "max": 179,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.calibrated_angular.counts_per_360_x",
                        "label": "精确角度控制：X 每圈 counts",
                        "type": "float",
                        "min": 1,
                        "step": 1,
                        "precision": 0,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.calibrated_angular.counts_per_360_y",
                        "label": "精确角度控制：Y 每圈 counts",
                        "type": "float",
                        "min": 1,
                        "step": 1,
                        "precision": 0,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.calibrated_angular.kp_x",
                        "label": "精确角度控制：Kp X",
                        "type": "float",
                        "min": 0,
                        "max": 2,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.calibrated_angular.kp_y",
                        "label": "精确角度控制：Kp Y",
                        "type": "float",
                        "min": 0,
                        "max": 2,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.calibrated_angular.kd_x",
                        "label": "精确角度控制：Kd X",
                        "type": "float",
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.calibrated_angular.kd_y",
                        "label": "精确角度控制：Kd Y",
                        "type": "float",
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.calibrated_angular.d_ema_alpha",
                        "label": "精确角度控制：D 项 EMA",
                        "type": "float",
                        "min": 0.01,
                        "max": 1,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.calibrated_angular.max_angle_step_x_deg",
                        "label": "精确角度控制：X 最大角度步长",
                        "type": "float",
                        "min": 0.0001,
                        "step": 0.0001,
                        "precision": 4,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.calibrated_angular.max_angle_step_y_deg",
                        "label": "精确角度控制：Y 最大角度步长",
                        "type": "float",
                        "min": 0.0001,
                        "step": 0.0001,
                        "precision": 4,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.universal_saturated.response_scale_x_px",
                        "label": "通用控制：水平响应尺度",
                        "type": "float",
                        "min": 0.1,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.universal_saturated.response_scale_y_px",
                        "label": "通用控制：垂直响应尺度",
                        "type": "float",
                        "min": 0.1,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.universal_saturated.max_step_x_counts",
                        "label": "通用控制：最大水平移动",
                        "type": "float",
                        "min": 0.1,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.universal_saturated.max_step_y_counts",
                        "label": "通用控制：最大垂直移动",
                        "type": "float",
                        "min": 0.1,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.freshness_threshold_ms",
                        "label": "稳健预测控制：控制新鲜度 ms",
                        "type": "float",
                        "min": 1,
                        "max": 200,
                        "step": 1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.projection.fov_x_deg",
                        "label": "稳健预测控制：水平 FOVX",
                        "type": "float",
                        "min": 30,
                        "max": 179,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.projection.counts_per_360",
                        "label": "稳健预测控制：每圈 counts",
                        "type": "float",
                        "min": 1,
                        "step": 1,
                        "precision": 0,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.mode.near_threshold_px",
                        "label": "稳健预测控制：NEAR 阈值 px",
                        "type": "float",
                        "min": 0,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.velocity.smoothing_frames",
                        "label": "稳健预测控制：速度平滑帧数",
                        "type": "float",
                        "min": 0.1,
                        "max": 20,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.velocity.history_reset_gap_ms",
                        "label": "稳健预测控制：历史中断重置 ms",
                        "type": "float",
                        "min": 0.1,
                        "max": 500,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.velocity.spread_base_px_ms",
                        "label": "稳健预测控制：速度离散基础容许 px/ms",
                        "type": "float",
                        "min": 0.001,
                        "step": 0.001,
                        "precision": 3,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.velocity.spread_relative",
                        "label": "稳健预测控制：速度离散相对容许",
                        "type": "float",
                        "min": 0,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.velocity.change_base_px_ms",
                        "label": "稳健预测控制：趋势变化基础容许 px/ms",
                        "type": "float",
                        "min": 0.001,
                        "step": 0.001,
                        "precision": 3,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.velocity.change_relative",
                        "label": "稳健预测控制：趋势变化相对容许",
                        "type": "float",
                        "min": 0,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.prediction.lead_frames",
                        "label": "稳健预测控制：前瞻帧数",
                        "type": "float",
                        "min": 0,
                        "max": 10,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.prediction.far.absolute_cap_px",
                        "label": "稳健预测控制：FAR 预测绝对上限",
                        "type": "float",
                        "min": 0,
                        "max": 100,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.prediction.far.base_cap_px",
                        "label": "稳健预测控制：FAR 预测基础上限",
                        "type": "float",
                        "min": 0,
                        "max": 100,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.prediction.far.relative_cap",
                        "label": "稳健预测控制：FAR 预测相对上限",
                        "type": "float",
                        "min": 0,
                        "max": 2,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.prediction.near.absolute_cap_px",
                        "label": "稳健预测控制：NEAR 预测绝对上限",
                        "type": "float",
                        "min": 0,
                        "max": 100,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.prediction.near.base_cap_px",
                        "label": "稳健预测控制：NEAR 预测基础上限",
                        "type": "float",
                        "min": 0,
                        "max": 100,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.prediction.near.relative_cap",
                        "label": "稳健预测控制：NEAR 预测相对上限",
                        "type": "float",
                        "min": 0,
                        "max": 2,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.atan.far.kp",
                        "label": "稳健预测控制：FAR Kp",
                        "type": "float",
                        "min": 0.001,
                        "max": 0.999,
                        "step": 0.001,
                        "precision": 3,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.atan.near.kp",
                        "label": "稳健预测控制：NEAR Kp",
                        "type": "float",
                        "min": 0.001,
                        "max": 0.999,
                        "step": 0.001,
                        "precision": 3,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.atan.scale_counts",
                        "label": "稳健预测控制：共享 Atan 尺度",
                        "type": "float",
                        "min": 0.1,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.atan.far.max_counts_per_update",
                        "label": "稳健预测控制：FAR 单次上限",
                        "type": "float",
                        "min": 0.1,
                        "max": 2000,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.algorithms.dual_phase_atan_robust_predictive_v2.atan.near.max_counts_per_update",
                        "label": "稳健预测控制：NEAR 单次上限",
                        "type": "float",
                        "min": 0.1,
                        "max": 2000,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.shared.deadzone_x_px",
                        "label": "共享：X 到位阈值",
                        "type": "float",
                        "min": 0,
                        "max": 10,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.shared.deadzone_y_px",
                        "label": "共享：Y 到位阈值",
                        "type": "float",
                        "min": 0,
                        "max": 10,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.shared.max_count_slew_x",
                        "label": "共享：X counts 增长限制",
                        "type": "float",
                        "min": 0.1,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.shared.max_count_slew_y",
                        "label": "共享：Y counts 增长限制",
                        "type": "float",
                        "min": 0.1,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.shared.trigger_activation_delay_ms",
                        "label": "共享：触发后启动延迟 ms",
                        "type": "float",
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "precision": 0,
                        "restart_required": False,
                    },
                    {
                        "path": "control.recoil.enabled",
                        "label": "启用独立压枪",
                        "type": "bool",
                        "restart_required": False,
                    },
                    {
                        "path": "control.recoil.startup_ms",
                        "label": "压枪启动斜坡 ms",
                        "type": "float",
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "precision": 0,
                        "restart_required": False,
                    },
                    {
                        "path": "control.recoil.base_rate_counts_s",
                        "label": "压枪基础速率 counts/s",
                        "type": "float",
                        "min": 0,
                        "max": 20000,
                        "step": 1,
                        "precision": 0,
                        "restart_required": False,
                    },
                    *[
                        {
                            "path": path,
                            "label": label,
                            "type": "float",
                            "min": minimum,
                            "max": maximum,
                            "step": step,
                            "precision": precision,
                            "restart_required": False,
                        }
                        for path, label, minimum, maximum, step, precision in (
                            ("control.recoil.max_rate_counts_s", "压枪最大速率 counts/s", 0, 20000, 1, 0),
                            ("control.recoil.positive_deadzone_norm", "压枪保持范围", 0, 1, 0.01, 2),
                            ("control.recoil.negative_deadzone_norm", "压枪刹车范围", 0, 1, 0.01, 2),
                            ("control.recoil.full_brake_error_norm", "完全刹车误差范围", 0, 1, 0.01, 2),
                            ("control.recoil.fast_add_gain_counts_s", "压枪追加强度 counts/s", 0, 20000, 1, 0),
                            ("control.recoil.max_fast_add_ratio", "追加强度上限比例", 0, 1, 0.01, 2),
                            ("control.recoil.stale_threshold_ms", "压枪数据过期阈值 ms", 0, 5000, 1, 0),
                        )
                    ],
                    {
                        "path": "control.output_enabled",
                        "label": "发送偏移控制量",
                        "type": "bool",
                        "restart_required": False,
                    },
                    {
                        "path": "control.scheduler_enabled",
                        "label": "启用 Scheduler 分步发送",
                        "type": "bool",
                        "restart_required": False,
                    },
                    {
                        "path": "control.scheduler_step_counts_x",
                        "label": "Scheduler X 单步 counts",
                        "type": "int",
                        "min": 1,
                        "max": 20,
                        "step": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.scheduler_step_counts_y",
                        "label": "Scheduler Y 单步 counts",
                        "type": "int",
                        "min": 1,
                        "max": 20,
                        "step": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.scheduler_interval_ms",
                        "label": "Scheduler 间隔 ms",
                        "type": "float",
                        "min": 1,
                        "max": 10,
                        "step": 0.1,
                        "precision": 1,
                        "restart_required": False,
                    },
                    {
                        "path": "control.target_fov_radius_px",
                        "label": "目标选择半径（640 基准 px）",
                        "type": "float",
                        "min": 1,
                        "step": 1,
                        "precision": 0,
                        "restart_required": False,
                    },
                    {
                        "path": "control.candidate_selection_class_weight",
                        "label": "目标分数：类别权重",
                        "type": "float",
                        "min": 0,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
                    {
                        "path": "control.candidate_selection_distance_weight",
                        "label": "目标分数：距离权重",
                        "type": "float",
                        "min": 0,
                        "step": 0.01,
                        "precision": 2,
                        "restart_required": False,
                    },
                    {
                        "path": "control.target_switch_delay_ms",
                        "label": "目标切换延迟 ms",
                        "type": "float",
                        "min": 0,
                        "max": 500,
                        "step": 1,
                        "precision": 0,
                        "restart_required": False,
                    },
                    {
                        "path": "control.trigger_mode",
                        "label": "触发方式",
                        "type": "select",
                        "options": ["hardware", "always"],
                        "restart_required": False,
                    },
                ],
            },
            {
                "id": "hardware",
                "label": "硬件盒子",
                "fields": [
                    {
                        "path": "hardware.auto_connect",
                        "label": "服务启动自动连接",
                        "type": "bool",
                        "restart_required": False,
                    },
                    {
                        "path": "hardware.host",
                        "label": "盒子地址",
                        "type": "string",
                        "restart_required": True,
                    },
                    {
                        "path": "hardware.port",
                        "label": "盒子端口",
                        "type": "int",
                        "min": 0,
                        "max": 65535,
                        "restart_required": True,
                    },
                    {
                        "path": "hardware.uuid",
                        "label": "kmNet UUID",
                        "type": "string",
                        "restart_required": True,
                    },
                    {
                        "path": "hardware.monitor_port",
                        "label": "kmNet 监听端口",
                        "type": "int",
                        "min": 0,
                        "max": 65535,
                        "restart_required": True,
                    },
                ],
            },
            {
                "id": "logging",
                "label": "日志",
                "fields": [
                    {
                        "path": "logging.level",
                        "label": "日志级别",
                        "type": "select",
                        "options": ["DEBUG", "INFO", "WARNING", "ERROR"],
                        "restart_required": False,
                    },
                    {
                        "path": "logging.dir",
                        "label": "日志目录",
                        "type": "string",
                        "restart_required": False,
                    },
                ],
            },
        ],
    }
    schema["sections"] = _split_control_sections(schema["sections"])
    for section in schema["sections"]:
        for field in section["fields"]:
            spec = param_schema_for(str(field.get("path", "")))
            if spec:
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


def _split_control_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for section in sections:
        if section.get("id") != "control":
            result.append(section)
            continue

        fields = list(section.get("fields", []))
        by_path = {str(field.get("path", "")): field for field in fields}
        selector = by_path.get("control.active_algorithm")
        if selector is not None:
            selector["option_labels"] = {
                algorithm_id: algorithm_definition(algorithm_id).product_name
                for algorithm_id in supported_algorithm_ids()
            }

        prefixes = {
            UNIVERSAL_SATURATED: "control.algorithms.universal_saturated.",
            CALIBRATED_ANGULAR: "control.algorithms.calibrated_angular.",
            DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2: (
                "control.algorithms.dual_phase_atan_robust_predictive_v2."
            ),
        }
        algorithm_fields = {
            algorithm_id: [
                field
                for field in fields
                if str(field.get("path", "")).startswith(prefix)
            ]
            for algorithm_id, prefix in prefixes.items()
        }
        algorithm_paths = {
            str(field.get("path", ""))
            for scoped_fields in algorithm_fields.values()
            for field in scoped_fields
        }
        common_control_paths = {
            "control.shared.trigger_activation_delay_ms",
            "control.aim.role_y_ratios.head",
            "control.aim.role_y_ratios.body",
            "control.aim.role_y_ratios.other",
        }
        non_predictive_common_paths = {
            "control.configured_actuation_delay_s",
        }
        non_predictive_common_fields = [
            field
            for field in fields
            if str(field.get("path", "")) in non_predictive_common_paths
        ]
        standard_output_prefixes = ("control.shared.", "control.scheduler_")
        standard_output_fields = [
            field
            for field in fields
            if str(field.get("path", "")).startswith(standard_output_prefixes)
            and str(field.get("path", "")) not in common_control_paths
        ]
        standard_output_paths = {
            str(field.get("path", "")) for field in standard_output_fields
        }
        shared_fields = [
            field
            for field in fields
            if str(field.get("path", "")) not in algorithm_paths
            and str(field.get("path", "")) not in non_predictive_common_paths
            and str(field.get("path", "")) not in standard_output_paths
        ]
        common_scope = list(supported_algorithm_ids())
        result.extend(
            [
                {
                    "id": "control_shared",
                    "label": "控制公共设置",
                    "algorithm_scope": common_scope,
                    "fields": shared_fields,
                },
                {
                    "id": "control_universal_saturated",
                    "label": "通用控制",
                    "algorithm_scope": [UNIVERSAL_SATURATED],
                    "fields": algorithm_fields[UNIVERSAL_SATURATED],
                },
                {
                    "id": "control_calibrated_angular",
                    "label": "精确角度控制",
                    "algorithm_scope": [CALIBRATED_ANGULAR],
                    "fields": algorithm_fields[CALIBRATED_ANGULAR],
                },
                {
                    "id": "control_robust_predictive",
                    "label": "稳健预测控制",
                    "algorithm_scope": [DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2],
                    "fields": algorithm_fields[DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2],
                },
                {
                    "id": "control_non_predictive_shared",
                    "label": "通用/精确公共设置",
                    "algorithm_scope": [UNIVERSAL_SATURATED, CALIBRATED_ANGULAR],
                    "fields": non_predictive_common_fields,
                },
                {
                    "id": "control_standard_output",
                    "label": "通用/精确输出策略",
                    "algorithm_scope": [UNIVERSAL_SATURATED, CALIBRATED_ANGULAR],
                    "fields": standard_output_fields,
                },
            ]
        )
    return result


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
