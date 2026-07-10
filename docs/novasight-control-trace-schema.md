# NovaSight Control Trace Schema

文档状态：trace 字段说明  
schema：`novasight.control_trace` version `1`  
格式：JSONL，每行一个 control observation trace record

## 范围

本 schema 只记录控制运行链路的只读指标：

```text
DetectionBatch
-> Tracker / Kalman
-> control calculation
-> Scheduler
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
- `Scheduler created/expires`：monotonic ns。
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
| counts planned/queued/sent/estimated_applied/unobserved | counts |
| scheduler pending_age | ms |
| timestamp fields | ns |

`tracker.velocity_px_s` in schema v1 is a legacy field name. Its meaning is observed/estimated screen-space line-of-sight velocity, not target-world velocity. The `predictive_pid_v2` path will use explicit `raw_observed_*` and `filtered_observed_*` names in its next trace version.

## Correlation ID

单次控制观察使用：

```text
control:{detection_generation}:{frame_id}:{capture_ts_ns}
```

该 ID 串联：

- DetectionBatch generation/frame/capture
- control calculation
- Scheduler trajectory_generation / command_id
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

## 采集位置

- Runtime 在已有 recording hook 中调用 `build_control_trace_record()`。
- Tracker/Kalman 指标来自 `last_control.pipeline.estimated_target_state` 和 `track_diagnostics`。
- Angular Controller 指标来自 `last_control.pipeline.angular_controller` 及控制 debug payload。
- Scheduler 指标来自 execution metadata 中的 `scheduler` 或 `scheduler.status()`。
- Device send 时间来自 `ExecutorRegistry` 包裹实际 executor 调用时记录的 monotonic start/end。

## 示例

测试生成的示例见：

```text
docs/examples/control_trace_v1.jsonl
```

该示例不是 Jetson 实测数据，只用于固定 schema 和序列化形态。
