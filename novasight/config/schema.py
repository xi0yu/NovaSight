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
                        "options": ["gst_cpu_latest", "nvmm_latest", "deepstream_nvinfer"],
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
                    {"path": "calibration.game_sensitivity_fingerprint", "label": "灵敏度指纹", "type": "string", "restart_required": False},
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
                        "options": ["tensorrt", "nvmm_latest", "deepstream_nvinfer"],
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
                    {"path": "inference.deepstream_io_mode", "label": "DeepStream V4L2 IO 模式", "type": "int", "min": 0, "max": 5, "restart_required": True},
                    {"path": "inference.deepstream_batched_push_timeout_us", "label": "DeepStream 批次等待 us", "type": "int", "min": 0, "restart_required": True},
                    {"path": "inference.deepstream_parser_library", "label": "DeepStream C++ Parser", "type": "string", "restart_required": True},
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
                    {"path": "control.active_algorithm", "label": "控制算法", "type": "select", "options": ["universal_saturated", "calibrated_angular", "ttbox_pid_atan", "dual_phase_atan_predictive_v1"], "restart_required": False},
                    {"path": "control.aim.y_ratio", "label": "瞄点纵向比例", "type": "float", "min": 0, "max": 1, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.configured_actuation_delay_s", "label": "估计执行延迟 s", "type": "float", "min": 0, "max": 0.1, "step": 0.001, "precision": 3, "restart_required": False},
                    {"path": "control.prediction_strength", "label": "预测强度", "type": "float", "min": 0, "max": 1.5, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.prediction_x_enabled", "label": "预测 X", "type": "bool", "restart_required": False},
                    {"path": "control.prediction_y_enabled", "label": "预测 Y", "type": "bool", "restart_required": False},
                    {"path": "control.algorithms.calibrated_angular.fov_x_deg", "label": "精确标定：水平 FOVX", "type": "float", "min": 30, "max": 179, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.calibrated_angular.counts_per_360_x", "label": "精确标定：X 每圈 counts", "type": "float", "min": 1, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "control.algorithms.calibrated_angular.counts_per_360_y", "label": "精确标定：Y 每圈 counts", "type": "float", "min": 1, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "control.algorithms.calibrated_angular.kp_x", "label": "精确标定：Kp X", "type": "float", "min": 0, "max": 2, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.algorithms.calibrated_angular.kp_y", "label": "精确标定：Kp Y", "type": "float", "min": 0, "max": 2, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.algorithms.calibrated_angular.kd_x", "label": "精确标定：Kd X", "type": "float", "min": 0, "max": 1, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.algorithms.calibrated_angular.kd_y", "label": "精确标定：Kd Y", "type": "float", "min": 0, "max": 1, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.algorithms.calibrated_angular.d_ema_alpha", "label": "精确标定：D 项 EMA", "type": "float", "min": 0.01, "max": 1, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.algorithms.calibrated_angular.max_angle_step_x_deg", "label": "精确标定：X 最大角度步长", "type": "float", "min": 0.0001, "step": 0.0001, "precision": 4, "restart_required": False},
                    {"path": "control.algorithms.calibrated_angular.max_angle_step_y_deg", "label": "精确标定：Y 最大角度步长", "type": "float", "min": 0.0001, "step": 0.0001, "precision": 4, "restart_required": False},
                    {"path": "control.algorithms.universal_saturated.response_scale_x_px", "label": "通用适配：水平响应尺度", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.universal_saturated.response_scale_y_px", "label": "通用适配：垂直响应尺度", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.universal_saturated.max_step_x_counts", "label": "通用适配：最大水平移动", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.universal_saturated.max_step_y_counts", "label": "通用适配：最大垂直移动", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.ttbox_pid_atan.response_scale_x_px", "label": "ttbox_pid_atan：水平响应尺度 px", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.ttbox_pid_atan.response_scale_y_px", "label": "ttbox_pid_atan：垂直响应尺度 px", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.ttbox_pid_atan.per_frame_gain_x", "label": "ttbox_pid_atan：X 每帧增益", "type": "float", "min": 0.001, "step": 0.001, "precision": 3, "restart_required": False},
                    {"path": "control.algorithms.ttbox_pid_atan.per_frame_gain_y", "label": "ttbox_pid_atan：Y 每帧增益", "type": "float", "min": 0.001, "step": 0.001, "precision": 3, "restart_required": False},
                    {"path": "control.algorithms.ttbox_pid_atan.max_step_x_counts", "label": "ttbox_pid_atan：X 最大 counts", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.ttbox_pid_atan.max_step_y_counts", "label": "ttbox_pid_atan：Y 最大 counts", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.ttbox_pid_atan.normalize_to_dt", "label": "ttbox_pid_atan：按 dt 归一", "type": "bool", "restart_required": False},
                    {"path": "control.algorithms.ttbox_pid_atan.nominal_dt_s", "label": "ttbox_pid_atan：标称帧间隔 s", "type": "float", "min": 0.001, "max": 0.2, "step": 0.001, "precision": 3, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.freshness_threshold_ms", "label": "双阶段 v1：控制新鲜度 ms", "type": "float", "min": 1, "max": 200, "step": 1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.aim.y_ratio", "label": "双阶段 v1：瞄点纵向比例", "type": "float", "min": 0, "max": 1, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.projection.fov_x_deg", "label": "双阶段 v1：水平 FOVX", "type": "float", "min": 30, "max": 179, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.projection.counts_per_360", "label": "双阶段 v1：每圈 counts", "type": "float", "min": 1, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.projection.invert_y", "label": "双阶段 v1：反转 Y", "type": "bool", "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.mode.near_enter_min_px", "label": "双阶段 v1：进入 NEAR 最小 px", "type": "float", "min": 0, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.mode.near_exit_min_px", "label": "双阶段 v1：退出 NEAR 最小 px", "type": "float", "min": 0, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.mode.near_enter_bbox_h_ratio", "label": "双阶段 v1：进入 NEAR 框高比", "type": "float", "min": 0, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.mode.near_exit_bbox_h_ratio", "label": "双阶段 v1：退出 NEAR 框高比", "type": "float", "min": 0, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.far.kp", "label": "双阶段 v1：FAR Kp", "type": "float", "min": 0.001, "max": 0.999, "step": 0.001, "precision": 3, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.far.atan_scale_counts", "label": "双阶段 v1：FAR Atan 尺度", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.far.max_counts_per_update", "label": "双阶段 v1：FAR 单次上限", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.near.kp", "label": "双阶段 v1：NEAR Kp", "type": "float", "min": 0.001, "max": 0.999, "step": 0.001, "precision": 3, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.near.atan_scale_counts", "label": "双阶段 v1：NEAR Atan 尺度", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.near.max_counts_per_update", "label": "双阶段 v1：NEAR 单次上限", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.estimator.measurement_std_px", "label": "双阶段 v1：测量噪声 px", "type": "float", "min": 0.01, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.estimator.acceleration_std_px_s2", "label": "双阶段 v1：加速度噪声", "type": "float", "min": 0.1, "step": 1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.estimator.min_dt_ms", "label": "双阶段 v1：最小 dt ms", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.estimator.reset_dt_ms", "label": "双阶段 v1：重置 dt ms", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.prediction.enabled_x", "label": "双阶段 v1：启用 X 预测", "type": "bool", "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.prediction.enabled_y", "label": "双阶段 v1：启用 Y 预测（v1 固定关闭）", "type": "bool", "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.prediction.actuation_delay_ms", "label": "双阶段 v1：执行延迟 ms", "type": "float", "min": 0, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.prediction.max_horizon_ms", "label": "双阶段 v1：最大预测时域 ms", "type": "float", "min": 0, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.prediction.far_weight", "label": "双阶段 v1：FAR 预测权重", "type": "float", "min": 0, "max": 1, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.prediction.near_weight", "label": "双阶段 v1：NEAR 预测权重", "type": "float", "min": 0, "max": 1, "step": 0.01, "precision": 2, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.prediction.far_abs_cap_px", "label": "双阶段 v1：FAR 预测绝对上限", "type": "float", "min": 0, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.algorithms.dual_phase_atan_predictive_v1.prediction.near_abs_cap_px", "label": "双阶段 v1：NEAR 预测绝对上限", "type": "float", "min": 0, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.shared.deadzone_x_px", "label": "共享：X 到位阈值", "type": "float", "min": 0, "max": 10, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.shared.deadzone_y_px", "label": "共享：Y 到位阈值", "type": "float", "min": 0, "max": 10, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.shared.max_count_slew_x", "label": "共享：X counts 增长限制", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.shared.max_count_slew_y", "label": "共享：Y counts 增长限制", "type": "float", "min": 0.1, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.shared.invert_y", "label": "共享：反转 Y 轴", "type": "bool", "restart_required": False},
                    {"path": "control.shared.trigger_activation_delay_ms", "label": "共享：触发后启动延迟 ms", "type": "float", "min": 0, "max": 1000, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "control.shared.recoil_enabled", "label": "共享：启用 Y 轴压枪", "type": "bool", "restart_required": False},
                    {"path": "control.shared.recoil_start_delay_ms", "label": "共享：压枪启动延迟 ms", "type": "float", "min": 0, "max": 1000, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "control.shared.recoil_y_rate_counts_s", "label": "共享：Y 压枪速率 counts/s", "type": "float", "min": 0, "max": 5000, "step": 1, "precision": 1, "restart_required": False},
                    {"path": "control.shared.recoil_ramp_up_ms", "label": "共享：压枪渐入 ms", "type": "float", "min": 0, "max": 2000, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "control.shared.recoil_max_counts_per_observation", "label": "共享：单观测最大压枪 counts", "type": "float", "min": 0.1, "max": 20, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.scheduler_enabled", "label": "启用 Scheduler 分步发送", "type": "bool", "restart_required": False},
                    {"path": "control.scheduler_step_counts_x", "label": "Scheduler X 单步 counts", "type": "int", "min": 1, "max": 20, "step": 1, "restart_required": False},
                    {"path": "control.scheduler_step_counts_y", "label": "Scheduler Y 单步 counts", "type": "int", "min": 1, "max": 20, "step": 1, "restart_required": False},
                    {"path": "control.scheduler_interval_ms", "label": "Scheduler 间隔 ms", "type": "float", "min": 1, "max": 10, "step": 0.1, "precision": 1, "restart_required": False},
                    {"path": "control.target_fov_radius_px", "label": "目标选择半径 px", "type": "float", "min": 1, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "control.target_switch_delay_ms", "label": "目标切换延迟 ms", "type": "float", "min": 0, "max": 500, "step": 1, "precision": 0, "restart_required": False},
                    {"path": "control.trigger_mode", "label": "触发方式", "type": "select", "options": ["hardware", "always"], "restart_required": False},
                ],
            },
            {
                "id": "hardware",
                "label": "硬件盒子",
                "fields": [
                    {"path": "hardware.auto_connect", "label": "服务启动自动连接", "type": "bool", "restart_required": False},
                    {"path": "hardware.host", "label": "盒子地址", "type": "string", "restart_required": True},
                    {"path": "hardware.port", "label": "盒子端口", "type": "int", "min": 0, "max": 65535, "restart_required": True},
                    {"path": "hardware.uuid", "label": "kmNet UUID", "type": "string", "restart_required": True},
                    {"path": "hardware.monitor_port", "label": "kmNet 监听端口", "type": "int", "min": 0, "max": 65535, "restart_required": True},
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
