# NovaSight TODO

> 架构基调：Jetson-first，纯双机模式，React + FastAPI，视觉分析算法和控制算法插件化。采集、推理、控制通过清晰边界连接，优先保障真机低延迟和可诊断性。
>
> 算法整改基准：严格执行 `/Users/zhangxiaoyu/Downloads/NovaSight-算法系统整改设计书-简体中文版.md`。生产控制主链固定为 `error_px -> error_rad -> angular PD -> calibrated counts -> scheduler -> DeviceAdapter`，禁止像素域经验倍率绕过该链路。

## 当前策略

- 主分支策略：主要开发只使用 `develop-alpha`，不从 `main` 合并。
- 采集策略：不能固定单一输入规格。后端必须枚举 `/dev/video0` 支持的全部格式、分辨率、帧率，前端展示并允许用户切换。
- 预览策略：后端输出 MJPEG 预览流，前端用原生 `<img>` 拉流。后续检测框、FOV、轨迹由后端绘制到原始帧后再编码。
- 控制策略：只保留双机路径。移除本地 Win32 API、单机截图、本地全局热键等单机依赖。
- 队列策略：采集和推理之间使用覆盖式单帧队列，容量为 1，永远处理最新帧。
- 容错策略：核心采集和推理控制线程 Fail-Fast。未捕获异常写日志并退出进程，拒绝带病运行。
- 测试策略：阶段实现完成后集中补关键路径测试。减少单个覆盖面小的测试，不追求 100%，目标约 80% 且覆盖真机关键风险。
- UI 策略：前端改为中文采集工作台，强调设备能力、配置切换、实时画面、诊断状态，不保留英文控制台文案。
- 整改执行策略：以 `TODO.md` 作为唯一主看板。每次按“可运行的大模块”推进，避免散落计划文件、频繁琐碎改动和新增过窄测试文件。

## 已完成基础

- [x] Python 项目结构和 `pyproject.toml`
- [x] FastAPI 应用骨架
- [x] Vite + React + TypeScript 前端骨架
- [x] 模型注册、发布和基础推理运行边界
- [x] 视觉分析插件和控制插件基础结构
- [x] 控制输出模式：静默、命令行、dry-run、kmNet 占位
- [x] V4L2 能力解析：`v4l2-ctl --list-formats-ext`
- [x] 采集 profile 自动选择：高帧率、低延迟、均衡、手动
- [x] 采集诊断 CLI：`doctor camera`、`capture-smoke`
- [x] 采集状态 API：`/api/capture/state`
- [x] 采集能力 API：`/api/capture/capabilities`
- [x] 采集选择 API：`/api/capture/select`

## Phase 1：中文采集工作台

- [x] 后端增加 MJPEG 预览流接口。
- [x] 后端在流启动时支持按当前配置自动打开采集源。
- [x] 后端支持“前端选择能力列表项后重启采集源”。
- [x] 后端输出可诊断的采集状态：设备、后端、格式、分辨率、帧率、采集等待、丢帧、恢复次数、最近错误。
- [x] 前端展示 `/dev/video0` 支持的全部格式列表。
- [x] 前端提供格式、分辨率、帧率筛选和一键应用。
- [x] 前端 Live View 使用 MJPEG `<img>` 显示真实画面。
- [x] 前端中文化：导航、状态、空态、错误、按钮、表格、模型和插件区域文案。
- [x] 前端 UI/UX 优化为“采集配置工作台”：配置优先、画面可见、诊断紧凑。
- [x] Jetson 验证文档：后端启动、前端启动、能力枚举、切换配置、预览流检查。

## Phase 2：运行流水线

- [x] 实现 `LatestFrameQueue`，容量严格为 1，写入覆盖旧帧。
- [x] 建立 Thread A 采集循环，读取最新配置快照并写入单帧队列。
- [x] 建立 Thread B 推理和控制循环，从单帧队列读取最新帧。
- [x] 增加线程安全配置快照，避免帧中途参数突变。
- [x] 增加 WebSocket 状态推送，频率限制在 10Hz 到 20Hz。
- [x] 将运行状态拆为采集、推理、控制、硬件盒子四个诊断块。

## Phase 3：推理闭环

- [x] TensorRT 作为 Jetson 主路径，保留 ONNX/不可用 runtime 作为开发边界。
- [x] 模型加载预热。
- [x] 标准化检测结果输出：坐标、类别、置信度、帧 ID、时间戳。
- [x] 后端在预览帧上绘制检测框、FOV 和轨迹。
- [x] 推理异常进入 Fail-Fast 路径。

## Phase 4：硬件盒子与控制

- [x] 定义硬件盒子接口：连接、移动、点击、读取物理输入状态。
- [x] 实现 kmNet 网络适配器。
- [x] 预留 MAKCU 串口适配器。
- [x] 解析盒子回传的物理按键状态，作为唯一触发源。
- [x] 增加盒子心跳保护，断连时挂起追踪逻辑。
- [x] 定义控制策略接口。
- [x] 实现 PID 控制策略，包含积分限幅和微分低通。
- [x] 实现预测控制策略。
- [x] 增加控制指令节流和合并，避免超过盒子接收极限。

## Phase 5：日志、异常和联调

- [x] 建立日志目录和轮转策略。
- [x] 核心线程入口统一包裹异常处理。
- [x] 崩溃前写入 crash 日志，并尽量通过 WebSocket 推送 `FATAL_ERROR`。
- [x] 实现 Fail-Fast 退出。
- [x] 增加坐标映射验证：绘制框和 1080p 原始像素坐标一致。
- [ ] 真机验证：端到端延迟基准，目标移动到盒子输出命令，目标小于 30ms。
- [ ] 真机验证：拔线测试，采集卡、盒子网线或串口断开后能正确退出或挂起。

## Phase 6：算法系统整改主线

目标：把现有“实验算法 + 运行时补丁”收敛成设计书定义的生产控制链路。每个模块必须有清晰 interface，调用方不再理解内部细节。

### 6.1 几何与角度控制链路

- [x] 新增 `novasight/control/angular.py`，包含 `CalibrationProfile`、`AngularErrorMapper`、`AngularPDController`。
- [x] `experimental_angle_pid` 改为使用 `AngularErrorMapper -> AngularPDController`，不再在 strategy 内散算焦距、atan、counts。
- [x] 禁止 ROI 尺寸作为焦距 fallback。缺少完整 control/capture 尺寸时返回 `CONTROL_GEOMETRY_INVALID`。
- [x] ROI 目标点先转换为完整控制坐标：`comp_x = roi_offset_x + aim_x`，`comp_y = roi_offset_y + aim_y`。
- [x] Y 轴误差在角度映射层使用图像坐标：`error_y_px = comp_y - center_y`。
- [x] 增加 counts residual，避免小于 1 count 的控制量被每帧吞掉。
- [ ] 把 `fov_x_deg`、`counts_per_360_x/y`、`axis_sign_x/y` 从普通控制参数迁移为持久化 Calibration Profile。
- [ ] 配置变更时如果 FOV、counts 或轴方向变化，强制清空 AngularPD/Kalman/调度状态。

### 6.2 生产命令调度

- [x] 新增生产 `CommandScheduler`：节流窗口内只保留最新命令，不再合并旧命令惯性。
- [x] `ExecutorRegistry` 接入 `CommandScheduler`，策略输出先经过 policy/y limiter，再由 scheduler 决定是否发送。
- [x] Scheduler 状态快照进入 executor status，UI/诊断可见 pending、节流窗口、取消计数。
- [ ] Scheduler 增加完整命令 TTL、过期取消日志和运行时统计。
- [ ] 新检测/新 frame 到来时取消旧 pending 命令，日志记录 cancel reason。
- [ ] 调度层负责拆分大 counts 和 move_auto/bezier 时长约束，DeviceAdapter 不重新解释角度或目标误差。

### 6.3 方向与设备边界

- [ ] 消除 `KmNetExecutor.flip_dy` 对生产控制链路的二次方向翻转。
- [ ] 保留一个兼容开关用于旧算法迁移，但 `experimental_angle_pid` 必须只走 Calibration Profile 的 `axis_sign_y`。
- [ ] kmNet 执行日志同时输出 requested counts、calibrated counts、driver counts，便于定位方向错位。

### 6.4 跟踪、预测与时间语义

- [x] `capture_ts_ns` 随 `FrameContext` 进入策略，Kalman 使用 capture timestamp 计算新观测 dt。
- [x] `InferenceThread` 观测更新与 `ControlThread` 高频 tick 已分离，推理帧不直接发 HID。
- [ ] 抽出 `DetectionBatch`、`TrackState`、`EstimatedState`、`CompensatedTarget` 数据契约，替代 strategy raw dict。
- [ ] Kalman 增加 NIS、协方差阈值、missing_ms 和 prediction_confidence，过度外推必须停控。
- [ ] 目标切换必须走 `SWITCH_COMMITTED`，只有提交切换后才重置 Kalman/PID/EMA。

### 6.5 日志、回放与验收

- [ ] 日志记录每次控制的 `error_px`、`error_rad`、`u_rad`、float counts、residual、final counts、driver counts。
- [ ] 增加回放入口，用记录文件复现 CandidateFilter -> Tracker -> AngularPD -> Scheduler。
- [ ] 增加静态检查：生产路径不得出现 `Kp * error_px -> DeviceAdapter` 的可达路径。
- [ ] 真机验收：固定目标收敛、移动目标预测、漏检三帧内持续控制、FOV/counts 变更后停控。

## 测试清理策略

- [x] 清理绑定旧架构、旧接口、旧模型管理方式的测试。
- [x] 合并过窄测试，保留覆盖真实风险的关键路径测试。
- [x] 阶段功能完成后集中补 `pytest`：
  - 配置加载与严格校验
  - 采集能力解析和配置选择
  - 采集切换 API
  - MJPEG 流基本行为
  - Latest-Frame Queue 覆盖语义
  - Fail-Fast 边界和日志记录
  - 控制输出限幅和静默模式
