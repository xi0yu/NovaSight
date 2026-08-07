# NovaSight 开火首发延迟与运行时热更新设计

- 日期：2026-08-06
- 状态：设计已获口头确认，待用户审阅规格
- 范围：开火首发延迟的前后端协同实现；说明并约束现有 manifest 模型切换行为

## 1. 背景与目标

NovaSight 已支持部分参数在运行中的主链内热更新，但当前没有独立的开火首发延迟参数。已有的 `pipeline.target_switch_delay_ms`、`pipeline.actuation_feedback_delay_ms`、`hardware.trigger_poll_interval_ms` 和 `control.recoil.interval_ms` 分别属于目标切换、设备反馈、硬件轮询和压枪节奏，不能替代开火延迟。

本次目标：

1. 增加 `pipeline.fire_delay_ms`，表达硬件扳机按下后的首个物理输出等待时间。
2. 在 Studio“参数设置”调整该值。
3. 前端、HTTP API、配置事务、RuntimeSupervisor 和 PipelineRuntime 完成同一条热更新闭环。
4. 运行中的配置调整不重启 `novasightd`，不停止采集、推理、跟踪或控制线程。
5. 扳机松开时 fail-closed，不能发送等待期间或此前排队的旧命令。
6. 解释并保持 manifest 模型切换的真实边界：当前不重启 `novasightd` 进程，但会停止并重建运行时推理主链；本次不将模型双管线热切换混入开火延迟范围。

## 2. 非目标

- 不把开火延迟并入压枪间隔、视觉反馈等待或硬件轮询间隔。
- 不改变 `control.trigger_mode` 的 `always` / `hardware` 语义。
- 不在本次改动中实现真正的模型双管线热切换。
- 不改变 manifest 的格式、fingerprint 算法、inspect/profile/probe/publish 顺序。
- 不为需要进程级资源重建的配置伪造“无需重启”状态。

## 3. 用户可见语义

新增字段：

```text
pipeline.fire_delay_ms
```

默认值为 `0 ms`，范围为 `0..=1000 ms`，单位为毫秒。默认值保证升级后不改变已有行为。

当 `control.trigger_mode = hardware` 时：

- 扳机从未按下变为按下：记录本次触发周期的开始时间，并等待 `fire_delay_ms`。
- 延迟期间继续运行采集、推理、目标选择、跟踪和控制计算，但不产生设备副作用。
- 延迟到期后，只允许最新 generation 且通过既有输出门、设备连接、触发状态、反馈等待和 epoch 校验的命令发送。
- 扳机松开立即取消等待、清空 command slot、清除控制遥测和待发送状态。
- 如果参数在等待期间被修改，新值只影响下一次扳机上升沿；当前周期使用上升沿时读取的截止时间，避免拖动参数导致当前等待跳变。

当 `control.trigger_mode = always` 时，没有硬件扳机上升沿，该字段不参与输出门控。切换回 `hardware` 后，按硬件触发语义生效。

Studio 控件显示“即时生效”，并明确说明：

> 硬件扳机按下后等待该时间再发送首个物理控制量；松开立即取消。

延迟等待期间的运行状态使用明确的阻断码 `FIRE_DELAY_PENDING`，并显示剩余毫秒，避免把“配置已保存”误认为“运行时已生效”。

## 4. 方案与决策

### 方案 A：在 AimAlgorithm 内延迟

让算法状态机观察触发上升沿并延迟产生 `AimResult`。改动集中，但算法并不拥有最终设备发送权；已有 command slot 可能在算法之外排队，难以保证延迟和松开都不泄漏旧命令。

### 方案 B：复用 `control.recoil.interval_ms`

可以快速复用配置和前端控件，但会把首发时序与压枪节奏耦合，无法表达“瞄准移动首次输出也等待、压枪仍按自己的间隔”这两个不同概念。

### 方案 C：在 PipelineRuntime 的设备输出门前延迟（采用）

设备 worker 是物理副作用的最终门控点。在该处处理触发上升沿、截止时间和 command slot，可阻断旧命令，并复用现有 `device_lane` 锁、`set_trigger_active(false)` 清理、`output_gate_min_generation` 和 epoch 校验。热更新只扩展 `PipelineLiveConfig`，不重建任何 worker。

## 5. 后端架构与数据流

### 5.1 配置模型与 schema

在 `crates/novasight-config/src/model.rs` 的 `PipelineRuntimeConfig` 增加 typed 字段、默认值和统一范围校验；`bootstrap.yaml` 写入显式默认值；repository 的 canonical YAML 映射同步该字段，避免 YAML、typed config 和 runtime config 的含义分叉。

在 `crates/novasight-api/src/dto/config_schema.rs` 的实时 `pipeline` section 增加：

```text
path: pipeline.fire_delay_ms
label: 开火首发延迟
type: float
min: 0
max: 1000
unit: ms
restart_required: false
```

schema 标记和 Rust 校验必须使用同一范围。字段必须进入 schema 热字段契约测试。

### 5.2 Runtime 配置发布

保持现有 actor 事务与 desired/effective revision 语义：

```text
Studio 控件
  → persistRuntimeConfigField
  → POST /api/config {section:"pipeline", key:"fire_delay_ms", value}
  → apply_config_field_update
  → ConfigService::update_pipeline
  → RuntimeCommand::UpdatePipelineConfig
  → ConfigService::persist_pipeline
  → RuntimeDependencies::install_live_pipeline_config
  → PipelineIngress::set_live_config
```

`PipelineLiveConfig` 增加 `fire_delay_ns`。`set_live_config` 在已有 `device_lane` 锁保护下校验并发布该值；配置版本发布后清除 command slot，使新的参数边界不会让旧命令泄漏。持久化失败或运行时安装失败时，返回错误并保持旧运行时配置。

无需新增第二套配置保存协议。`/api/v1/config`、`novasight-client` 和 CLI 继续使用现有字段更新契约；如需要 CLI 暴露便捷参数，调用既有 `update_config_field`。

### 5.3 触发时序

`SharedState` 增加热发布的 `fire_delay_ns`，以及设备 worker 独占的当前触发周期状态：

- `trigger_started_at_ns: Option<u64>`
- `trigger_deadline_ns: Option<u64>`
- `trigger_delay_pending: bool`

触发状态的读写遵守以下规则：

1. 硬件触发从 false 变 true 时，在 device worker 观察到上升沿的当前单调时钟上记录开始时间，并以当前 `fire_delay_ns` 计算固定 deadline。
2. 在 deadline 之前保留 command slot 中最新的 `OutputPlan`，不发送也不丢弃；设备 worker 使用 `LatestSlot::wait_take_or_timeout` 等待新 plan 或 deadline 到期。新 plan 到达时替换旧待发 plan，避免积累多个 move。
3. deadline 到期后，取当前最新 plan，继续执行现有发送前检查；只发送最新 generation。若 deadline 到期时没有 plan，继续等待下一条最新 plan，不创建独立补发任务。
4. false 状态优先级最高：清空周期状态、清空 command slot、清除控制遥测和当前待发 plan。
5. 非硬件模式不建立等待周期。
6. `fire_delay_ns = 0` 立即通过现有输出检查，不增加等待。
7. 配置热更新不修改已有周期 deadline；下一次上升沿读取新值。

这里的等待只限制设备副作用，不阻塞上游实时链。不会调用 `stop_state`、不会关闭 DeepStream、不会清理 tracker，也不会创建独立定时器补发命令；`wait_take_or_timeout` 只是设备 worker 已有 slot 消费循环的超时等待。

### 5.4 运行时状态与 trace

`fire_delay_pending` 与剩余时间是设备 worker 产生的运行时事实，需要通过 `PipelineMetrics`/`RuntimeSnapshot` 发布；不能仅由 API 根据 `trigger_active` 和配置值推断。为避免 status 读取阻塞设备 worker，发布使用已有轻量原子/快照机制。

为验证端到端生效，在 runtime control/prediction 状态中增加：

- `fire_delay_ms`
- `fire_delay_pending`
- `fire_delay_remaining_ms`

新增阻断原因 `FIRE_DELAY_PENDING`，映射到已有 runtime status / control trace DTO 与前端文案。字段没有值时使用 `null`，而不是伪造 `0`；非硬件模式可报告 `pending=false`。

状态是诊断信息，不作为设备 worker 的第二套决策源。设备 worker 的 deadline 是唯一输出门控事实。

## 6. 前端实现

### 6.1 API 与类型

`web/src/api.ts` 现有 `RuntimeConfigValue`、`ConfigUpdateResponse` 和 `updateRuntimeConfigField` 已能承载该字段，无需新增不一致的 endpoint。补充必要的状态类型字段，使返回的 runtime trace 可安全读取。

### 6.2 参数模型

`web/src/features/studio/algorithmParameterModel.ts`：

- 将 `fire_delay_ms` 加入 `CONTROL_PIPELINE_FIELDS`。
- 在 `AlgorithmParameterValues` 增加对应值。
- 在控制参数组中增加参数元数据：范围 `0..1000`、单位 `ms`、`applyMode: "live"`、高级风险等级。
- 更新 schema 一致性测试，确保字段存在、数值类型正确且 `restart_required=false`。

### 6.3 Studio 控件

`web/src/features/studio/StudioConsoleView.tsx` 在“控制模式”区域加入 `ParameterNumberControl`。提交调用：

```ts
updateConfigField("pipeline", "fire_delay_ms", value)
```

沿用现有：

- 乐观 draft
- `configWriteQueueRef` 串行提交
- expected revision CAS
- 成功后 canonical revision 更新
- 失败后 draft 回滚与错误提示
- 后端 schema 回传刷新

不新增前端绕过 `runtimeConfigPersistence.ts` 的写路径。

### 6.4 状态文案

`web/src/features/studio/controlTrace.ts` 与共享 runtime 状态映射新增 `FIRE_DELAY_PENDING`；延迟期间显示“正在等待开火首发延迟”和剩余时间。配置响应 `applied=true` 且 `restart_required=false` 时显示即时生效；若后端返回重启要求，前端不得将其标为 live。

## 7. Manifest 模型切换边界

manifest 是模型运行契约与验证产物，不是独立的模型热替换器。当前流程：

```text
inspect → configure/profile → probe → 原子写入 <engine>.manifest.json → publish
```

manifest 提供：

- Engine 真实 I/O Shape、Batch、DataType
- 预处理语义
- 输出 Tensor 与 parser/decoder 契约
- 类别、objectness、阈值和 NMS 参数
- Engine fingerprint 与校验 receipt

publish 由 `RuntimeSupervisor::activate_model_state` 执行：

```text
校验候选 manifest/receipt
  → preflight 新模型
  → 如运行中则 stop_state
  → SQLite active deployment 提交
  → 安装新 model geometry/contract
  → start_state 重建运行时感知和推理管线
  → 失败时补偿 catalog、geometry 和旧运行时
```

所以当前准确语义是：

- 不重启 `novasightd` 进程。
- 运行中的模型切换会停止并重建 PipelineRuntime / DeepStream 感知主链。
- 旧 frame、目标和控制状态不会跨模型继续执行。
- 本次只确保参数热更新不触发该流程；不修改模型切换提示或伪称其为真正无停机热切换。

真正不停止主链的模型切换需要双 Engine/双 pipeline、候选 warm-up、DetectionBatch readiness、active ingress 原子切换、generation 隔离、旧资源延迟释放和失败回退，作为独立设计处理。

## 8. 错误处理与安全性

- 任何非法范围、非有限值或 schema 不匹配都在保存前拒绝。
- revision 冲突返回现有配置冲突错误，前端保留服务器新值并提示刷新。
- 持久化成功但 live install 失败时不报告成功；保留旧 runtime 值，并按现有配置事务错误路径处理 desired/effective 差异。
- 扳机松开、output gate 关闭、切入 hardware 模式和 runtime stop 都必须 fail-closed，清空 command slot。
- 延迟期间不创建补发定时器，不积累多个待发 move，不跨 epoch 发送。
- 开火延迟只增加等待，不改变现有 freshness、feedback、device connection 和 output gate 保护。

## 9. 验证计划

遵循 TDD：每个新行为先写失败测试，确认失败原因是缺少 `fire_delay` 行为，再写最小实现。

### Rust 配置与 schema

- 默认值、序列化/反序列化和 YAML 映射。
- 0 与 1000 合法；负数、超范围、NaN、Infinity 拒绝。
- schema 字段存在且 `restart_required=false`。
- pipeline 更新返回 `applied=true`、`restart_required=false`。

### PipelineRuntime 行为

使用现有 fake clock / fake pointer device 约定：

- 硬件触发上升沿后 deadline 前无设备发送。
- deadline 到期后发送最新 command。
- 扳机松开立即取消等待且无残留发送。
- 上升沿前 command 不会在 deadline 后泄漏。
- 等待中更新配置不改变当前 deadline。
- 下一次上升沿使用新配置。
- `fire_delay_ms=0` 保持原即时发送行为。
- `always` 模式不受 fire delay 影响。
- 热更新过程中不产生 pipeline stop/start 事件。

### API 与前端

- Studio 控件读取 schema/current value，并以 live 标签显示。
- 提交使用 `/api/config` 的 pipeline field update。
- 成功响应更新 revision、draft 和 runtime config。
- 失败响应回滚 draft 并显示错误。
- trace/status 的 `FIRE_DELAY_PENDING` 和剩余时间正确展示。

### Manifest 回归

- 保持 inspect/profile/probe/publish receipt 校验。
- 保持模型切换当前 stop/start 行为和失败补偿。
- 前端继续明确显示“运行中切换会停止并重启推理主链”，不将其误报为无停机模型热切换。
