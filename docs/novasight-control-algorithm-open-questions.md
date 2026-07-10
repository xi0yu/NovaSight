# NovaSight 控制算法开放问题与实验清单

文档状态：开放问题产物  
范围：只记录算法审计中无法由当前数学和代码证据完全关闭的问题  
规则：需要真实 Jetson、HID/KMBOX、游戏画面或运行数据的问题标记为实验项或阻塞项，不用推测填空

## 1. 实验项

### E-01 prediction horizon 的真实组成

- 类别：实验项
- 缺少什么：Jetson 实测 trace，包含 `capture_ts_ns`、`inference_end_ts_ns`、`control_now_ns`、Scheduler emit、device send、画面反馈。
- 为什么当前不能验证：当前代码能记录 frame age 和部分 pipeline timing，但无法证明 HID/KMBOX 到游戏画面反馈的真实延迟。
- 最小验证步骤：录制 60 秒运行 trace，人工或脚本标注命令发送与画面响应。
- 指标：`configured_extra_prediction_delay_ms`、`prediction_horizon_ms`、`device_send_to_visual_feedback_ms`、`result_age_ms`、`overshoot_px`。

### E-02 Kp、Kd 与 prediction_gain 的耦合参数

- 类别：实验项
- 缺少什么：动态目标 A/B 测试。
- 为什么当前不能验证：数学能证明预测与 D 项耦合，但不能决定具体参数。
- 最小验证步骤：固定模型、FOV、counts 标定，分别测试 `P only`、`prediction+P`、`prediction+small-D`。
- 指标：首帧接近时间、中心附近 RMS error、过冲次数、方向反转恢复时间。

### E-03 near 区域 D 项策略

- 类别：实验项
- 缺少什么：中心附近微调数据。
- 为什么当前不能验证：不同游戏灵敏度、目标速度和检测抖动会改变最优阻尼。
- 最小验证步骤：对静态目标和匀速目标分别测试 near Kp/Kd scale。
- 指标：center crossing count、deadzone exit count、micro jitter counts、settling time。

### E-04 动态死区是否应跟 bbox 高度联动

- 类别：实验项
- 缺少什么：不同距离目标的检测框抖动统计。
- 为什么当前不能验证：bbox 高度与画面距离、模型抖动、目标部位选择有关。
- 最小验证步骤：按 bbox 高度分桶记录静态目标 bbox center 抖动。
- 指标：bbox_h_px、center_jitter_p95_px、deadzone_px、false_move_rate。

### E-05 最小速度观测帧数

- 类别：实验项
- 缺少什么：目标出现后前 2 到 5 帧的速度可信度统计。
- 为什么当前不能验证：当前默认 `min_velocity_measurements=3` 是保守选择，不是硬件验证结论。
- 最小验证步骤：A/B 测试 2、3、4 帧启动预测。
- 指标：initial_overshoot_px、first_lock_ms、velocity_nis、prediction_confidence。

### E-06 Y 轴是否关闭预测或使用更弱预测

- 类别：实验项
- 缺少什么：不同游戏目标垂直运动与 recoil/视角反馈数据。
- 为什么当前不能验证：Y 轴可能受目标姿态、枪械后坐力、bbox 高度变化影响，不能套用 X 轴。
- 最小验证步骤：分别测试 X/Y 同等预测、Y 弱预测、Y 无预测。
- 指标：vertical_overshoot_px、head_anchor_jitter_px、dy_counts_rms。

### E-07 最小有效 counts 与小数 residual

- 类别：实验项
- 缺少什么：设备和游戏对小 counts 的响应曲线。
- 为什么当前不能验证：整数 counts 与游戏灵敏度存在非线性/死区可能。
- 最小验证步骤：发送固定 counts 阶梯，测量画面角度变化。
- 指标：min_effective_counts_x/y、counts_to_rad_slope、hysteresis、residual_accumulation_error。

### E-08 counts_per_360 与 axis sign 标定

- 类别：实验项
- 缺少什么：当前游戏、分辨率、灵敏度下的标定记录。
- 为什么当前不能验证：配置默认值只能保证代码可运行，不能证明真实角度映射准确。
- 最小验证步骤：执行 360 度或固定角度标定流程，记录返回误差。
- 指标：counts_per_360_x/y、axis_sign_x/y、calibration_error_deg、repeatability。

### E-09 Scheduler TTL 与分步大小

- 类别：实验项
- 缺少什么：真实 HID/KMBOX 发送频率和游戏响应。
- 为什么当前不能验证：代码有 TTL 和 max step，但最佳值取决于设备和游戏帧。
- 最小验证步骤：测试不同 TTL、max_step、min_interval。
- 指标：pending_age_ms、cancelled_pending、expired_pending、device_error、overshoot。

### E-10 目标切换阈值

- 类别：实验项
- 缺少什么：多目标场景数据。
- 为什么当前不能验证：当前代码有 sticky/advantage/confirm_frames 机制，但阈值需要场景验证。
- 最小验证步骤：构造双目标交叉、遮挡、分数波动场景。
- 指标：false_switch_count、switch_latency_frames、identity_uncertain_duration。

### E-11 近期自身控制对屏幕速度可信度的影响

- 类别：实验项
- 缺少什么：静止目标、不同方向和不同强度实际设备输出下的屏幕位移 trace。
- 为什么当前不能验证：屏幕视线速度包含目标相对运动、camera-induced motion 和检测噪声，且发送到画面反馈存在未知延迟。
- 最小验证步骤：记录成功发送 counts 的 20/40/60 ms 窗口，运行静止目标无输出、静止目标单向输出、同向跟随和左右摆动四类轨迹。
- 指标：`raw_observed_vx_px_s`、`filtered_observed_vx_px_s`、`executed_counts_last_20ms`、`executed_counts_last_40ms`、`executed_counts_last_60ms`、`velocity_confidence`、`center_crossing_count`。

## 2. 阻塞项

### B-01 已发送但未生效 counts 无法精确估计

- 类别：阻塞项
- 缺少什么：设备发送 ACK 时间、游戏画面反馈时间，或外部观测回放。
- 影响：无法可靠计算 `theta_unobserved_control`，只能使用简化延迟模型。
- 最小解除条件：记录 `sent_ts_ns`、`counts`、下一帧画面变化，建立 applied/unobserved 估算。

### B-02 游戏反馈延迟未知

- 类别：阻塞项
- 缺少什么：从 HID/KMBOX 到游戏相机变化的真实延迟。
- 影响：prediction horizon 只能部分包含 actuation delay，参数会漂。
- 最小解除条件：建立一个测试画面或录像分析流程，测量 send-to-visual latency。

### B-03 没有真实长期运行 trace

- 类别：阻塞项
- 缺少什么：正常负载、过载、目标丢失、目标切换、UI 压力下的统一 trace。
- 影响：不能关闭 stale ratio、result age、control_observation_fps 和 pending 债务问题。
- 最小解除条件：至少采集每类 60 秒 trace，字段见主审计文档第 17 节。

### B-04 已执行 counts 与采集画面的时序对应未知

- 类别：阻塞项
- 缺少什么：Scheduler send、设备成功返回、游戏消费输入、画面产生和采集卡输出之间的统一时序标定。
- 影响：不能断言某个采集区间内发送的 counts 已经完整反映在当前帧中，因而不能把 self-motion subtraction 当正式真值。
- 最小解除条件：建立 send-to-visual trace，对不同负载下的延迟分布和抖动进行测量。

## 3. 待定实现项

这些问题不阻塞本轮算法审计，但需要后续独立任务决策。

### P-01 pending counts 是否进入控制器输入

- 类别：暂定结论
- 当前判断：需要，但必须先拆分 queued/sent/applied/unobserved。
- 风险：直接扣除 Scheduler pending 会把已取消或未发送命令也扣掉。
- 下一步：为 Runtime 增加控制执行状态快照，不在本轮修改代码。

### P-02 是否实现显式 Acquire/TrackMicro 状态

- 类别：暂定结论
- 当前判断：概念上需要；当前代码通过 tracker 状态、zone gain 和 Scheduler 替换已覆盖一部分。
- 风险：没有显式状态时，首次定位参数和微调参数耦合。
- 下一步：用 trace 证明当前连续 PD 是否已足够，再决定是否引入显式状态。

### P-03 EMA 使用固定 alpha 还是 tau

- 类别：暂定结论
- 当前判断：推理帧率不稳定时 tau 更合理。
- 风险：改动后参数含义变化，需要迁移配置。
- 下一步：先记录实际 `dt_s` 分布，再决定是否从 alpha 迁移到 tau。

### P-04 I 项是否彻底移除配置入口

- 类别：暂定结论
- 当前判断：当前算法不需要 I 项；`ki`/`integral_limit` 入口容易误导。
- 风险：删除配置会影响旧配置兼容；保留会让人误以为 PID 已实现。
- 下一步：单独做配置语义清理，不在本轮修改生产代码。

### P-05 D 项使用误差差分还是 Kalman 速度投影

- 类别：暂定结论
- 当前判断：Kalman 速度是屏幕视线速度估计，误差差分更像闭环阻尼；两者都包含自身控制影响，不能混用或称为纯目标速度。
- 风险：误差差分包含鼠标自身造成的画面运动。
- 下一步：同时记录 `observed_screen_velocity_rad_s`、`error_rate_rad_s` 与近期成功发送 counts，做 A/B。

### P-06 方向反转时 EMA/D 如何衰减

- 类别：暂定结论
- 当前判断：方向反转应快速清空或提高 alpha。
- 风险：过度清空会放大检测噪声。
- 下一步：加入 trace 字段 `error_sign_flip`，先评估反转频率。

### P-07 近中心最小输出策略

- 类别：暂定结论
- 当前判断：需要结合 deadzone、min effective counts 和 residual。
- 风险：最小输出过大会左右震荡；没有最小输出会卡在静态误差。
- 下一步：完成 E-07 后再定。

### P-08 自运动补偿何时可进入正式预测

- 类别：暂定结论
- 当前判断：阶段 3.5 只做 Shadow Mode；在 B-04 解除前不得使用 `estimated_relative_velocity` 驱动正式预测。
- 风险：错误的 counts-to-frame 对齐比不补偿更危险，会制造方向错误和预测尖峰。
- 下一步：对比 `observed_delta_px`、`estimated_self_delta_px`、`estimated_relative_delta_px` 与真实静止目标轨迹，再决定是否进入主线。

## 4. 本轮已关闭的问题

- EMA 不是控制器，不能替代 PD/PID。
- 两阶段控制不是二阶 PID。
- 当前 Angular 控制器不是完整 PID，I 项没有进入实际控制状态。
- 预测与 D 项存在速度补偿重复风险。
- stale DetectionBatch、generation 倒退、capture_ts 倒退不能进入控制。
- Scheduler 必须替换旧 pending，而不是累积旧命令。

## 5. 后续独立任务建议

1. 运行 trace schema 任务：把主审计文档第 17 节指标写入统一日志/录制格式。
2. pending/applied counts 建模任务：只处理 Scheduler、executor、Runtime 的状态闭环。
3. 自运动遥测任务：记录成功发送 counts 时间窗和 Shadow Mode self-induced displacement。
4. 控制参数 A/B 任务：固定硬件和模型后比较 P、prediction+P、prediction+small-D。
5. 标定任务：建立 counts_per_360、axis sign、min effective counts 的实测流程。
