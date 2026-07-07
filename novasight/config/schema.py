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
                        "options": ["cpu", "nvmm"],
                        "restart_required": True,
                    },
                    {"path": "capture.pixel_format", "label": "像素格式", "type": "string", "restart_required": True},
                    {"path": "capture.width", "label": "宽度", "type": "int", "min": 0, "restart_required": True},
                    {"path": "capture.height", "label": "高度", "type": "int", "min": 0, "restart_required": True},
                    {"path": "capture.fps", "label": "帧率", "type": "int", "min": 0, "restart_required": True},
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
                    {"path": "calibration.fov_x_deg", "label": "水平 FOVX", "type": "float", "min": 1, "max": 179, "restart_required": False},
                    {"path": "calibration.counts_per_360_x", "label": "水平每圈 counts", "type": "float", "min": 1, "restart_required": False},
                    {"path": "calibration.counts_per_360_y", "label": "垂直每圈 counts", "type": "float", "min": 1, "restart_required": False},
                    {"path": "calibration.axis_sign_x", "label": "X 轴方向", "type": "select", "options": ["-1", "1"], "restart_required": False},
                    {"path": "calibration.axis_sign_y", "label": "Y 轴方向", "type": "select", "options": ["-1", "1"], "restart_required": False},
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
                    {"path": "control.candidate_ratio_max_aspect", "label": "候选框最大宽高比", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.candidate_quality_confidence_weight", "label": "候选质量置信度权重", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.candidate_quality_area_weight", "label": "候选质量面积权重", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.class_priority_quality_margin", "label": "类别优先质量容忍", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.tracker_confirm_frames", "label": "Track 确认帧数", "type": "int", "min": 1, "max": 10, "restart_required": False},
                    {"path": "control.target_switch_min_preference_advantage", "label": "切换优势阈值", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.target_switch_min_continuity_score", "label": "切换连续性阈值", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.target_switch_confirm_frames", "label": "切换确认帧数", "type": "int", "min": 1, "max": 10, "restart_required": False},
                    {"path": "control.tracker_matching_distance_px", "label": "Track 匹配距离 px", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.tracker_ambiguity_margin", "label": "身份歧义边界", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.tracker_missing_timeout_ms", "label": "Track 漏检超时 ms", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.tracker_delete_timeout_ms", "label": "Track 删除超时 ms", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.tracker_match_threshold", "label": "Track 匹配代价上限", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.tracker_mahalanobis_gate", "label": "Track 马氏门控", "type": "float", "min": 0.001, "restart_required": False},
                    {"path": "control.kalman_enabled", "label": "启用 Kalman 估计", "type": "bool", "restart_required": False},
                    {"path": "control.kalman_acceleration_noise", "label": "Kalman 加速度噪声", "type": "float", "min": 0.001, "restart_required": False},
                    {"path": "control.kalman_measurement_noise_x", "label": "Kalman X 测量噪声", "type": "float", "min": 0.001, "restart_required": False},
                    {"path": "control.kalman_measurement_noise_y", "label": "Kalman Y 测量噪声", "type": "float", "min": 0.001, "restart_required": False},
                    {"path": "control.kalman_max_predict_missing_ms", "label": "Kalman 最大漏检预测 ms", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.kalman_max_predict_steps", "label": "Kalman 最大预测步数", "type": "int", "min": 1, "restart_required": False},
                    {"path": "control.kalman_max_predict_dt_ms", "label": "Kalman 单步最大 dt ms", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.kalman_max_position_sigma_px", "label": "Kalman 最大位置 sigma", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.kalman_max_covariance_trace", "label": "Kalman 最大协方差迹", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.kalman_nis_threshold", "label": "Kalman NIS 阈值", "type": "float", "min": 0.001, "restart_required": False},
                    {"path": "control.kalman_nis_hard_reject", "label": "Kalman NIS 硬拒收", "type": "float", "min": 0.001, "restart_required": False},
                    {"path": "control.kalman_min_identity_confidence", "label": "Kalman 最低身份可信度", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.kalman_min_prediction_confidence", "label": "Kalman 最低预测可信度", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.kalman_prediction_decay_tau_ms", "label": "Kalman 预测衰减 tau ms", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.aim_horizontal_percent", "label": "Aim 横向百分比", "type": "float", "min": 0, "max": 100, "restart_required": False},
                    {"path": "control.aim_ratio", "label": "瞄准高度", "type": "float", "min": 0, "max": 100, "restart_required": False},
                    {"path": "control.aim_offset_x_px", "label": "Aim X 偏移 px", "type": "float", "min": -200, "max": 200, "restart_required": False},
                    {"path": "control.aim_offset_y_px", "label": "Aim Y 偏移 px", "type": "float", "min": -200, "max": 200, "restart_required": False},
                    {"path": "control.aim_ema_enabled", "label": "Aim EMA 平滑", "type": "bool", "restart_required": False},
                    {"path": "control.aim_ema_alpha", "label": "Aim EMA Alpha", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.aim_max_anchor_jump_ratio", "label": "Aim 锚点跳变阈值", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.latency_compensation_enabled", "label": "启用延迟补偿", "type": "bool", "restart_required": False},
                    {"path": "control.latency_compensation_scale", "label": "延迟补偿比例", "type": "float", "min": 0, "max": 1.5, "restart_required": False},
                    {"path": "control.latency_max_compensation_ms", "label": "最大补偿时间 ms", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.latency_reject_if_age_exceeds_ms", "label": "测量过旧拒收 ms", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.latency_max_compensation_px", "label": "最大补偿距离 px", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.latency_min_velocity_px_s", "label": "补偿最低速度 px/s", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.latency_max_velocity_px_s", "label": "补偿最高速度 px/s", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.latency_min_velocity_measurements", "label": "补偿最少速度样本", "type": "int", "min": 1, "restart_required": False},
                    {"path": "control.latency_min_velocity_confidence", "label": "补偿最低速度可信度", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.latency_estimated_actuation_delay_ms", "label": "估计设备生效延迟 ms", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.strategy", "label": "鼠标算法", "type": "select", "options": ["experimental_angle_pid"], "restart_required": False},
                    {"path": "control.trigger_mode", "label": "触发方式", "type": "select", "options": ["hardware", "always"], "restart_required": False},
                    {"path": "control.command_interval_ms", "label": "命令步进间隔 ms", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.scheduler_command_ttl_ms", "label": "命令 TTL ms", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.scheduler_predicted_command_ttl_ms", "label": "预测命令 TTL ms", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.scheduler_cancel_on_new_frame", "label": "新帧取消旧命令", "type": "bool", "restart_required": False},
                    {"path": "control.scheduler_cancel_on_direction_change", "label": "方向反转取消旧命令", "type": "bool", "restart_required": False},
                    {"path": "control.scheduler_cancel_on_track_change", "label": "目标切换取消旧命令", "type": "bool", "restart_required": False},
                    {"path": "control.scheduler_max_step_x", "label": "Scheduler X单步上限", "type": "int", "min": 1, "restart_required": False},
                    {"path": "control.scheduler_max_step_y", "label": "Scheduler Y单步上限", "type": "int", "min": 1, "restart_required": False},
                    {"path": "control.scheduler_queue_hard_limit", "label": "Scheduler 队列硬上限", "type": "int", "min": 1, "restart_required": False},
                    {"path": "control.scheduler_device_error_cooldown_ms", "label": "设备错误冷却 ms", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.move_kind", "label": "移动方式", "type": "select", "options": ["raw", "enc_raw", "auto", "enc_auto", "bezier", "enc_bezier"], "restart_required": False},
                    {"path": "control.move_ms", "label": "移动铺展 ms", "type": "int", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_kp_x", "label": "实验角度PID Kp X", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_kp_y", "label": "实验角度PID Kp Y", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_ki", "label": "实验角度PID Ki", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_kd", "label": "实验角度PID Kd", "type": "float", "min": -1, "max": 1, "restart_required": False},
                    {"path": "control.experimental_angle_integral_limit", "label": "实验角度PID 积分限幅", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_config_level", "label": "实验角度PID 配置级别", "type": "select", "options": ["basic", "advanced", "developer"], "restart_required": False},
                    {"path": "control.experimental_angle_speed", "label": "实验角度PID 跟枪速度", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_smooth_factor", "label": "实验角度PID 平滑程度", "type": "float", "min": 0, "max": 0.95, "restart_required": False},
                    {"path": "control.experimental_angle_deadzone_px", "label": "实验角度PID 死区px", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_derivative_filter", "label": "实验角度PID D项滤波", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.experimental_angle_near_error_deg", "label": "实验角度PID 近区角误差°", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_far_error_deg", "label": "实验角度PID 远区角误差°", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_near_kp_scale", "label": "实验角度PID 近区Kp倍率", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_middle_kp_scale", "label": "实验角度PID 中区Kp倍率", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_far_kp_scale", "label": "实验角度PID 远区Kp倍率", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_near_kd_scale", "label": "实验角度PID 近区Kd倍率", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_middle_kd_scale", "label": "实验角度PID 中区Kd倍率", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_far_kd_scale", "label": "实验角度PID 远区Kd倍率", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_prediction_gain_min", "label": "实验角度PID 预测Kp最低倍率", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.experimental_angle_prediction_d_gain_min", "label": "实验角度PID 预测Kd最低倍率", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.experimental_angle_max_control_angle_deg", "label": "实验角度PID 最大角位移°", "type": "float", "min": 0.001, "restart_required": False},
                    {"path": "control.experimental_angle_max_step_counts", "label": "实验角度PID 单帧限幅", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.experimental_angle_max_counts_delta_x", "label": "实验角度PID X变化限幅", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.experimental_angle_max_counts_delta_y", "label": "实验角度PID Y变化限幅", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.experimental_angle_control_hz", "label": "实验角度PID 控制频率", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.experimental_angle_kalman_enabled", "label": "实验角度PID Kalman滤波", "type": "bool", "restart_required": False},
                    {"path": "control.experimental_angle_kalman_process_noise", "label": "实验角度PID Kalman过程噪声", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_kalman_measurement_noise", "label": "实验角度PID Kalman测量噪声", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_hungarian_enabled", "label": "实验角度PID 匈牙利匹配", "type": "bool", "restart_required": False},
                    {"path": "control.experimental_angle_matching_distance_px", "label": "实验角度PID 匹配距离", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.experimental_angle_max_extrapolate_frames", "label": "实验角度PID 最大外推帧", "type": "int", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_target_filter_enabled", "label": "实验角度PID 目标过滤", "type": "bool", "restart_required": False},
                    {"path": "control.experimental_angle_target_filter_min_score", "label": "实验角度PID 过滤置信度", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.experimental_angle_target_filter_fov_ratio", "label": "实验角度PID 过滤FOV", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.experimental_angle_target_filter_same_class", "label": "实验角度PID 同类过滤", "type": "bool", "restart_required": False},
                    {"path": "control.experimental_angle_prediction_lead_ms", "label": "实验角度PID 预测提前ms", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_extrapolate_confidence_decay", "label": "实验角度PID 外推置信衰减", "type": "float", "min": 0, "max": 1, "restart_required": False},
                    {"path": "control.experimental_angle_magnet_enabled", "label": "实验角度PID 磁性吸附", "type": "bool", "restart_required": False},
                    {"path": "control.experimental_angle_magnet_radius_px", "label": "实验角度PID 磁性半径", "type": "float", "min": 1, "restart_required": False},
                    {"path": "control.experimental_angle_magnet_strength", "label": "实验角度PID 磁性强度", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_magnet_curve", "label": "实验角度PID 磁性曲线", "type": "float", "min": 0.1, "restart_required": False},
                    {"path": "control.experimental_angle_magnet_deadzone_px", "label": "实验角度PID 磁性死区", "type": "float", "min": 0, "restart_required": False},
                    {"path": "control.experimental_angle_magnet_max_counts", "label": "实验角度PID 磁性限幅", "type": "float", "min": 0, "restart_required": False},
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
    for section in sections:
        for field in section.get("fields", []):
            path = str(field.get("path", ""))
            if "." not in path:
                continue
            root, key = path.split(".", 1)
            source_root = values.get(root)
            if not isinstance(source_root, dict) or key not in source_root:
                continue
            target_root = visible.setdefault(root, {})
            if isinstance(target_root, dict):
                target_root[key] = source_root[key]
    return visible
