# NovaSight 动态 PID Jetson 诊断记录

日期：2026-07-05

## 背景

同一套动态 PID 源码在别人自用项目中表现正常，但接入 NovaSight Jetson 链路后出现跟不上、过冲、震荡、乱飞或触发后体感不对的问题。

当前判断：问题不应优先归因于 PID.cpp 公式本身，而应优先怀疑 NovaSight 工作流传入 PID 前后的数据单位、时序、坐标系、输出单位与原项目不一致。

## 当前动态 PID 接入方式

NovaSight 当前给动态 PID 的关键入参是：

```text
current_error_x = aim_x - roi_center_x
current_error_y = roi_center_y - aim_y
recent_target_width = target.w
imgsize = roi_width
dt = capture_ts_ns 当前帧与上一帧差值，缺失时 fallback 为 monotonic / 120fps
```

动态 PID 输出：

```text
pid_output_x = control_loop(error_x, dt, target_width, roi_width)
pid_output_y = control_loop(error_y, dt, target_width, roi_width)
```

然后 NovaSight 将 PID 输出作为 kmNet 移动量交给执行层。

## 最高风险差异

### 1. current_error 单位可能不同

我们现在传的是 ROI 像素误差。

原项目可能传的是：

- 模型输入坐标误差，例如 320/640 内的误差
- 屏幕像素误差
- 已经换算过的鼠标 counts
- FOV/角度换算后的误差

如果单位不同，即使 PID 公式完全一致，Kp/Kd 等参数也完全不能通用。

### 2. PID 输出单位可能不同

NovaSight 当前把 `control_loop()` 返回值直接当作 kmNet move counts。

原项目可能还有额外换算：

- 鼠标 API 缩放
- 游戏灵敏度换算
- FOV/c360 换算
- 不同硬件驱动的移动单位

因此“PID 输出 = kmNet counts”是需要验证的假设，不能默认成立。

### 3. dt 可能不同

源码中 D 项强依赖：

```cpp
误差变化率 = (当前误差 - 上一次误差) / 时间间隔;
```

NovaSight 当前优先用采集时间戳差值。原项目可能使用：

- 固定 dt
- 推理循环 dt
- 控制循环实际 dt
- 帧间隔常量

dt 越小，D 项越容易被放大；dt 抖动，输出就会抖动。

### 4. 检测框抖动会被 D 项放大

检测框不是稳定传感器。模型输入、NMS、ROI 裁剪、目标选择、框映射任一处抖动，都会导致 error 每帧跳动。

D 项会放大这种跳动，表现为过冲、震荡或忽然反向移动。

### 5. Y 轴坐标系和 kmNet flip 可能叠加

NovaSight 当前动态 PID 的 Y 误差是：

```text
error_y = roi_center_y - aim_y
```

这表示向上为正。执行层 kmNet 还有 `flip_dy`。如果方向约定和 kmNet 实际方向不一致，会出现 Y 轴越修越偏、突然向上移动等问题。

### 6. 控制发送频率可能不同

原项目可能是稳定 100Hz / 144Hz / 240Hz 循环。

NovaSight 是采集、推理、控制联动，控制频率受推理输出和目标选择影响。如果控制频率不同，同一组 PID 参数也不会有相同手感。

## 需要记录的对齐数据

下一步不要先继续调参，应先记录以下字段，并与原项目同画面、同目标、同偏移情况下对比：

```text
frame_id
target_id / target_key
roi_size
target.x/y/w/h
aim_x / aim_y
current_error_x / current_error_y
dt_ms
recent_target_width
imgsize
P_x / I_x / D_x / output_x
P_y / I_y / D_y / output_y
kmNet driver_dx / driver_dy
kmNet move API
trigger_active
```

## 判断方法

### 情况 A：同样 50px 偏移，NovaSight 输出远大于原项目

说明输出单位或 Kp 参数对应单位错误。

### 情况 B：dt 在 8ms、20ms、40ms 之间跳

说明 D 项输入时序不稳定，应先验证固定 dt 或控制循环 dt。

### 情况 C：X 正常，Y 乱飞

优先检查：

- `error_y` 符号
- `hardware.flip_dy`
- kmNet move 的 Y 方向
- aim_y_ratio 是否落在目标框合理位置

### 情况 D：目标框看起来稳定，但 output 抖动

优先检查 D 项和 dt。

### 情况 E：目标框本身跳动

优先检查检测后处理、NMS、目标选择、ROI 映射，而不是 PID。

## 建议的下一步

1. 增加动态 PID 诊断日志或前端诊断面板。
2. 不新增算法分支，先只暴露实际输入输出值。
3. 对比原项目同场景下的：
   - error
   - dt
   - target_width
   - PID output
   - driver output
4. 根据差异再决定是修输入单位、修 dt、修 Y 方向，还是修 kmNet 输出层。

## 当前结论

动态 PID 公式只是链路中的一段。现在最需要验证的是：

```text
我们喂给 PID 的 current_error、dt、target_width、imgsize
是否和原作者实际喂进去的是同一种量。
```

如果这四个量不一致，源码复刻得再像，结果也不会一致。

## 2026-07-05 补充：单位边界修正

本次不再让动态 PID 直接吃 ROI 像素并直接输出 kmNet counts。新的动态 PID 链路统一为：

```text
bbox / aim point
  ↓
error_px = aim - roi_center
  ↓
error_rad = atan(error_px / focal_px)
  ↓
dynamic PID control_loop(error_rad, fixed_dt, target_width_rad, fov_rad)
  ↓
move_counts = pid_output_rad / (2π) * counts_per_revolution
  ↓
kmNet move(dx_counts, dy_counts)
```

关键约定：

- PID 输入单位：弧度误差 `rad`，不是 ROI 像素。
- PID 输出单位：弧度控制量 `rad`，不是 kmNet counts。
- kmNet 输出单位：最后一层统一转换成 counts。
- 时间单位：动态 PID 先使用固定控制周期 `dt = 1 / dynamic_pid_control_hz`，默认 60Hz。
- 目标宽度：传入 PID 的 `recent_target_width` 已从像素宽度换算为目标角宽度。
- 图像尺寸：传入 PID 的 `imgsize` 已从 ROI 宽度换算为 FOV 弧度。
- 检测框中心滤波：新增 `dynamic_pid_ema_alpha`，用于在进入 PID 前稳定 aim point。

因此，原 C++ PID 的状态机和控制循环仍保留，但它处理的是“角度域”的误差。对应地，`达标误差阈值` 和 `误差变化容限` 也从像素默认值改成角度默认值：

```text
dynamic_pid_target_error_threshold = 0.016 rad
dynamic_pid_error_change_tolerance = 0.012 rad
```

这两个值大致对应 640 ROI、105° FOV 下的 4px / 3px 量级。后续调参应围绕角度域，而不是直接套用别人像素域参数。
