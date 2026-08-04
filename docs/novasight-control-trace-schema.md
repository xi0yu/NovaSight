# NovaSight Control Trace Schema

文档状态：trace 字段说明  
schema：`novasight.control_trace` version `5`
格式：JSONL，每行一个 control observation trace record

## 范围

本 schema 只记录控制运行链路的只读指标：

```text
DetectionBatch
-> Tracker / Kalman
-> control calculation
-> algorithm-specific delivery (V2 latest-replace Scheduler or configured legacy delivery)
-> device send result
```

它不调整 `Kp`、`Kd`、`prediction_gain`、`deadzone`、Scheduler TTL 或任何控制参数。

## 顶层结构

```text
schema
field_units
correlation
detection
tracker
control
algorithm_decision
counts
scheduler
device
```

## 时钟语义

所有 timestamp 字段都使用对象格式：

```json
{"value": 1050000000, "clock_domain": "monotonic", "unit": "ns"}
```

当前控制链要求：

- `DetectionBatch.capture_ts_ns`：monotonic ns。
- `DetectionBatch.publish_ts_ns`：monotonic ns。
- `Tracker/Kalman state_ts_ns`：monotonic ns。
- `control_now_ts_ns`：monotonic ns。
- Scheduler `created/expires`：monotonic ns；仅无 Scheduler 的直发算法保持空值。
- `device_send_start/end_ts_ns`：monotonic ns。

GStreamer PTS 或 wall clock 不得直接与这些字段相减。

## 单位

每条 trace 都包含 `field_units`，核心单位如下：

| 字段 | 单位 |
|---|---|
| detection input/result/inference age | ms |
| tracker position | px |
| tracker screen-space velocity | px/s |
| control error_px | px |
| control error_rad | rad |
| control error_rate_rad_s | rad/s |
| P、D、U、prediction velocity term | rad |
| algorithm aim/real error/control error/prediction offset | px |
| algorithm velocity_x | px/s |
| V2 robust velocity segments/mean/median/selected velocity/spread | px/ms |
| algorithm full error/float demand/integer command/residual | counts |
| counts planned/queued/sent/estimated_applied/unobserved | counts |
| scheduler pending_age | ms |
| timestamp fields | ns |

`tracker.velocity_px_s` 是兼容字段名，含义是屏幕表观速度，不是目标世界速度。`algorithm_decision.estimator.velocity_x_px_s` 是 V1 Kalman 估计结果；V2 使用 `algorithm_decision.robust_velocity`，单位固定为 `px/ms`，两者不得混算。

## Correlation ID

单次控制观察使用：

```text
control:{detection_generation}:{frame_id}:{capture_ts_ns}
```

该 ID 串联：

- DetectionBatch generation/frame/capture
- control calculation
- Scheduler trajectory_generation / command_id（无 Scheduler 的直发算法为空）
- device send result

`correlation.capture_ts` 也使用 timestamp 对象格式，不能保存裸 `*_ns` 整数。

## Unknown 字段

真实设备反馈未接入前，以下字段必须保持 nullable/unknown：

```json
"estimated_applied_counts": {
  "x": null,
  "y": null,
  "status": "unknown",
  "reason": "device_feedback_unavailable"
}
```

同理：

```json
"unobserved_counts": {
  "x": null,
  "y": null,
  "status": "unknown",
  "reason": "device_feedback_unavailable"
}
```

禁止用猜测值填充这两个字段。

## algorithm_decision

版本 3 新增专用算法决策块。版本 4 为 `dual_phase_atan_robust_predictive_v2` 增加 measured error 和四点短窗速度；当前 V2 进一步记录毫秒级预测提前量与后坐力前馈。V2 至少记录：

```text
algorithm_id / phase / measurement_dt_ms
aim_px / bbox
error_measured_px / error_control_px
history position count / three segment velocities / mean / median / selected prediction velocity / spread / detection and track confidence
prediction reference dt / configured lead ms / raw and weighted offset / cap / safe offset
recoil mode / enabled / active / left hold / configured-per-observation / visual demand / requested / combined demand / emitted / residual / block reason
full_error_counts / float_demand / integer_command / quantizer_residual
overzero_detected / will_emit / block_reason / executor_success
delivery_mode / scheduler_used
```

`executor_success` 在控制计算完成、尚未调用设备时为 `null`；设备调用完成后由 Runtime 回写为实际发送结果。它不表示目标检测或控制计算是否有效。

该算法的固定发送语义为：

```json
{
  "delivery_mode": "latest_replace",
  "scheduler_used": true
}
```

`scheduler` 顶层仍为兼容结构。V2 写入 `used=true`、`delivery_mode=latest_replace`，并且最多暴露一条完整待发送命令；新观测覆盖旧命令，不形成分步轨迹。

## 采集位置

- Runtime 在已有 recording hook 中调用 `build_control_trace_record()`。
- Tracker/Kalman 指标来自 `last_control.pipeline.estimated_target_state` 和 `track_diagnostics`。
- Angular Controller 指标来自 `last_control.pipeline.angular_controller` 及控制 debug payload。
- 新算法的完整估计、预测、counts 和 latest-replace 语义来自 `last_control.pipeline`。
- Scheduler 指标来自 execution metadata 中的 `scheduler` 或 `scheduler.status()`。
- Device send 时间来自 `ExecutorRegistry` 包裹实际 executor 调用时记录的 monotonic start/end。

## 示例

测试生成的示例见：

```text
tests/test_control_trace.py
```

测试数据不是 Jetson 实测数据，只用于固定 schema 和序列化形态。
