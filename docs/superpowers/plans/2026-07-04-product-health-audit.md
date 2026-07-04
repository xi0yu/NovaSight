# NovaSight 产品健康审计与模块边界报告

日期：2026-07-04
分支：`develop-alpha`
目标：从“能跑”推进到“稳定、可解释、可长期维护的真实产品”。

## 1. 审计结论

NovaSight 当前已经具备产品骨架：

- Jetson 优先的 GStreamer 采集链路。
- ROI 帧输入到推理。
- ONNX / TensorRT 后缀自动选择。
- 目标选择、PID / 比例 / 预测控制。
- kmNet 驱动加载、连接、诊断移动。
- React 控制台、模型管理、授权许可、配置管理。

但它还没有达到健康产品的完整设计状态。最大问题不是某个函数写错，而是多个模块缺少统一的运行态治理：

- 配置更新没有完整事务语义。
- 运行态配置、用户期望配置、正在应用配置没有明确分层。
- 采集重建、模型切换、执行器重建都散落在路由层。
- 前端很多状态是从 runtime snapshot 推断，而不是后端给出明确应用结果。
- 采集、推理、控制、kmNet 的闭环虽然存在，但缺少统一的“为什么没有输出”的诊断模型。

本次已经修复了几处会真实破坏链路的问题，但这只是第一轮产品健康治理。下一阶段应建设统一的 `RuntimeReconfigurator`。

更新：`RuntimeReconfigurator` 已开始落地，`PUT /api/config` 的配置安装、ROI live rebuild、kmNet 连接恢复、配置保存和 pipeline auto-start 已经从路由层迁移到运行态 reconfiguration 模块。后续还需要继续把采集选择、模型切换和硬件重连纳入同一套报告模型。

## 2. 产品级设计原则

后续所有模块改造都应该遵循这些原则：

1. **失败不破坏旧状态**
   新配置、新模型、新采集链路、新执行器失败时，上一组可用链路必须继续可用。

2. **配置应用必须可解释**
   前端不能只收到 `200 OK` 或 `400 Bad Request`，而要知道哪些配置已生效、哪些回滚、哪些需要重启。

3. **运行态与编辑态分离**
   用户正在编辑的不等于当前运行中的。必须区分：
   - `desired_config`：用户希望保存的配置。
   - `pending_config`：正在尝试应用的配置。
   - `active_config`：当前真实运行配置。

4. **每个主模块都要有小接口和深实现**
   路由层不应该直接懂采集重建、模型切换、执行器恢复。它应该调用更深的模块接口。

5. **真实反馈优先**
   UI 里展示的 FPS、目标、控制量、kmNet move 次数，必须来自后端真实运行态。

6. **诊断链路必须闭环**
   当没有 move 时，用户应该能看到停在哪一层：
   采集无帧、推理无目标、目标未选中、触发未满足、控制量为 0、执行器未发送、kmNet 驱动失败。

## 3. 运行主链路

当前主链路：

```text
采集卡 / 图片源
  -> CaptureService
  -> CaptureSession
  -> CapturedFrame
  -> RuntimePipeline
  -> center_roi_frame
  -> InferenceRuntime
  -> Detection / TargetSelector
  -> ControlStrategy
  -> ControlIntent
  -> ExecutorRegistry
  -> kmNet / dry_run / console
  -> RuntimeState / WebSocket / MJPEG Preview
```

产品角度的关键判断：

- 采集是推理和控制的前置条件。
- ROI 是运行资源，不只是 UI 参数。
- 推理结果必须能还原到 ROI 坐标。
- 控制量必须基于同一套 ROI 坐标和目标选择结果。
- 执行器发送成功不等于物理移动成功，但至少必须暴露 driver 层返回和计数。

## 4. 模块健康分析

### 4.1 应用初始化与全局状态

位置：

- `novasight/api/app.py`

当前职责：

- 创建 FastAPI 应用。
- 初始化配置、模型仓库、执行器、硬件、采集、推理、运行服务。
- 加载 active model。
- 注册 license middleware 和 API routers。

边界评价：

- 作为 composition root 是合理的。
- 但 `app.state` 中的多个对象共享同一个配置对象引用，容易让路由层直接修改配置。
- `_load_active_model()` 只在启动时执行，后续模型切换由模型路由处理，两个入口的状态语义需要统一。

已发现风险：

- `app.state.config`、`runtime.config`、`runtime.config_store` 容易出现同步顺序问题。
- license middleware 只做入口阻断，不区分功能权限的细粒度行为。

产品化建议：

- 保留 `create_app()` 作为装配入口。
- 新增 `RuntimeContainer` 或 `RuntimeCoordinator` 管理 app.state 里的核心对象。
- 禁止业务路由直接多点修改 `app.state.config`。

### 4.2 配置模块

位置：

- `novasight/config/runtime.py`
- `novasight/config/schema.py`
- `novasight/api/routes_runtime.py`
- `novasight/runtime/config_store.py`

当前职责：

- 解析 YAML / JSON 配置。
- 严格拒绝未知 key。
- 校验 ROI、推理、控制、硬件参数。
- 输出配置 schema 给前端。
- `PUT /api/config` 应用配置。

边界评价：

- `runtime.py` 的 dataclass schema 是好方向。
- `schema.py` 能驱动前端配置页，这是产品化基础。
- 最大问题在 `routes_runtime.py`：路由层承担了配置安装、执行器重建、硬件重建、推理参数同步、ROI 重建、配置保存等过多职责。

已修复问题：

- ROI 热更新失败后不再保存失败配置。
- 配置失败后前端刷新后端真实配置。

仍需修复 / 改造：

- `PUT /api/config` 返回仍然只有 `restart_required`，表达力不足。
- `restart_required` 现在过于粗糙，无法表达“已热更新”“局部重建成功”“局部重建失败并回滚”。
- 配置应用没有统一 `prepare/apply/rollback/report` 接口。

目标接口建议：

```python
class RuntimeReconfigurator:
    def apply(self, desired: RuntimeConfig) -> ConfigApplyReport:
        ...
```

建议返回：

```json
{
  "accepted": true,
  "applied": true,
  "rolled_back": false,
  "active_config": {},
  "desired_config": {},
  "sections": [
    {
      "section": "roi",
      "impact": "live_capture_rebuild",
      "status": "applied",
      "message": "ROI pipeline rebuilt"
    }
  ]
}
```

### 4.3 采集模块

位置：

- `novasight/capture/service.py`
- `novasight/capture/session.py`
- `novasight/capture/source.py`
- `novasight/capture/pipeline.py`
- `novasight/capture/profile.py`
- `novasight/capture/caps.py`
- `novasight/api/routes_capture.py`

当前职责：

- 查询 v4l2 采集能力。
- 根据 profile 生成 GStreamer appsink pipeline。
- 打开 frame source。
- 维护 capture thread 和 latest frame。
- 产出 `CapturedFrame`。

边界评价：

- `CaptureSession` 是核心运行模块，接口较小。
- `CaptureService` 负责能力查询、配置和 session 管理，职责较多但仍可接受。
- `routes_capture.py` 仍直接承担配置保存和 runtime pipeline auto-start，后续应下沉。

已修复问题：

- 采集配置切换失败时不再直接破坏上一条可用链路。
- 打开新采集源失败时，会尽量恢复旧 profile。
- MJPEG 预览处理采集源重建后的 `frame_id` 回退。

当前风险：

- 采集源重建依赖同一个物理设备，旧 pipeline 已释放后，新旧都可能因为设备/驱动状态失败。
- `frame_id` 是 source 本地递增，采集源重建后从 1 开始，需要更正式的 generation 机制。
- image source 和 capture source 混在同一 CaptureService，需要明确 source kind。

建议改造：

- 为每次采集 session 增加 `capture_generation`。
- `CapturedFrame` 使用 `(generation, frame_id)` 作为唯一身份。
- `CaptureService.configure()` 返回结构化结果：

```json
{
  "applied": true,
  "previous_preserved": true,
  "profile": {},
  "backend": "gst-appsink:nvmm-mjpg-iomode2"
}
```

### 4.4 ROI 与帧资源模块

位置：

- `novasight/roi.py`
- `novasight/capture/pipeline.py`
- `novasight/runtime/service.py`
- `novasight/capture/preview.py`

当前职责：

- 计算中心 ROI。
- 根据 ROI 裁剪 frame。
- 将 ROI frame 传给推理。
- 将 detection 映射到预览。

边界评价：

- `roi.py` 已经是独立模块，这是正确方向。
- 但 ROI 同时影响采集 pipeline、推理输入、预览显示、控制坐标，当前缺少统一 `RoiFrame` / `FrameContext` 产品概念。

当前风险：

- ROI size 和模型输入 size 不是一回事，用户容易误解。
- GStreamer pipeline 可直接输出裁剪后的 ROI，但 runtime 仍可能再次调用 `center_roi_frame()`，需要确保不会重复裁剪或坐标错位。
- 前端预览用 ROI 尺寸显示，但推理模型可能自动 resize 到 256 / 320 / 640，必须用清晰文案解释。

产品化建议：

- 明确定义：

```text
CaptureFrame: 原始采集或采集 pipeline 输出帧
RoiFrame: 控制和推理共同消费的 ROI 视图
ModelInputFrame: 按模型输入 shape resize 后的张量输入
```

- Runtime status 中明确暴露：
  - `roi_frame.width/height`
  - `roi_frame.offset_x/y`
  - `model_input.width/height`
  - `detection_coordinate_space`

### 4.5 预览模块

位置：

- `novasight/api/routes_capture.py`
- `novasight/capture/preview.py`
- `web/src/features/studio/StudioConsoleView.tsx`

当前职责：

- MJPEG 推流。
- 后端绘制检测框。
- 前端显示实时画面。

边界评价：

- 后端绘制是正确方向，避免前端坐标重复计算。
- 前端历史上出现过重复绘制、颜色混淆问题，说明“后端画什么、前端画什么”的边界需要写死。

已修复问题：

- 预览流在采集源重建后不再永久等待旧 frame_id。

当前风险：

- 预览消费者和推理主链路共享同一个 capture session，需要持续保证预览限速不影响推理。
- 预览绘制依赖 runtime last context，如果帧不同步，必须明确显示“检测结果来自最近帧”而不是误导。

建议：

- 预览只显示后端绘制的模型检测结果。
- 前端不再重复画 detection box，除非是明确的调试层并有独立颜色和开关。
- 增加 preview diagnostic：
  - `preview_frame_id`
  - `runtime_context_frame_id`
  - `overlay_frame_delta`

### 4.6 推理模块

位置：

- `novasight/inference/runtime.py`
- `novasight/inference/onnxruntime_engine.py`
- `novasight/inference/tensorrt.py`
- `novasight/inference/input.py`

当前职责：

- 根据后缀选择 engine。
- 加载 ONNX / TensorRT。
- 将 ROI frame 转换为模型输入。
- 解码 YOLO 输出。

边界评价：

- `InferenceRuntime.prepare/commit` 是好的模型切换基础。
- TensorRT 和 ONNX Runtime 分开实现是合理边界。
- 但模型仓库、模型发布、推理运行状态之间还需要更清晰的产品闭环。

当前风险：

- ONNX 自动检测依赖 `onnx` 包，缺失时会跳过检测；这不应该影响 TensorRT，但日志容易让用户误解。
- TensorRT engine 的输入 shape 和模型版本 input_shape 可能不一致，必须以 engine 实际 binding 为准。
- YOLO 输出解码失败时，前端需要显示是“无目标”还是“解码失败”。

建议：

- 模型加载返回 `InferenceLoadReport`：
  - artifact path
  - backend
  - engine input shape
  - configured input shape
  - loaded classes
  - output shape
  - decode layout

- 前端模型推理页只给普通用户展示：
  - 当前运行模型
  - 后端
  - 输入尺寸
  - 置信度 / NMS / 类别过滤
  - 最近推理状态

工程调试信息放入折叠区。

### 4.7 模型仓库模块

位置：

- `novasight/model_registry/store.py`
- `novasight/model_registry/schema.py`
- `novasight/api/routes_models.py`
- `web/src/features/models/ModelsView.tsx`

当前职责：

- 管理项目、版本、产物、转换任务、部署状态。
- 支持 ONNX / engine ready artifact。
- 发布模型到 active deployment。

边界评价：

- SQLite registry 比硬编码模型路径健康。
- 但“项目/版本/产物”是工程概念，对普通用户偏重。
- 用户真正关心的是：有哪些模型、当前用哪个、能不能切、切失败为什么。

当前风险：

- active deployment 与 inference runtime 是两个状态，前端需要明确区分。
- artifact 所属 project/version 校验过严时，用户看到 “artifact does not belong to project” 不知道如何处理。
- `classes=1` 这类日志容易被误解为模型真实类别数量，需要 UI 解释类名来源。

建议产品表达：

- 普通用户视图：
  - 模型名称
  - 后端类型
  - 输入尺寸
  - 类别配置
  - 运行状态

- 工程调试视图：
  - project/version/artifact id
  - checksum
  - input_shape
  - conversion jobs

### 4.8 运行流水线模块

位置：

- `novasight/runtime/pipeline.py`
- `novasight/runtime/service.py`
- `novasight/runtime/status.py`
- `novasight/runtime/target_selector.py`

当前职责：

- 消费最新帧。
- 运行 ROI -> inference -> target selection -> control -> execution。
- 汇总状态给 WebSocket。

边界评价：

- 单线程 inference/control 是目前可接受设计。
- `RuntimeService.process_frame()` 职责过重，里面同时做推理、目标选择、控制量、执行状态记录。

已修复问题：

- 采集源重建后 `frame_id` 回退时，推理线程会重置消费游标。

当前风险：

- `RuntimeService.process_frame()` 是后续 bug 高发区，需要拆出深模块：
  - `InferenceStep`
  - `TargetSelectionStep`
  - `ControlStep`
  - `ExecutionStep`

建议：

- 增加 `RuntimeFrameTrace`，每帧记录：
  - frame id / generation
  - roi info
  - inference ran / reason
  - raw detections
  - selected target
  - control intent
  - execution result

这会让“为什么没有 move”成为可解释问题。

### 4.9 目标选择模块

位置：

- `novasight/runtime/target_selector.py`
- `novasight/runtime/service.py`

当前职责：

- 根据 FOV、类别优先级、目标锁定、粘性选择目标。

边界评价：

- 独立 target selector 是正确方向。
- 前端目前能显示 selection reason，这是健康产品必需项。

当前风险：

- 多目标时用户需要知道：
  - 候选目标数量
  - 被过滤原因
  - 最终目标序号
  - 是否保持锁定

建议：

- Runtime status 暴露 `target_candidates_debug`。
- 前端用折叠列表展示每个候选的 score、class、distance、inside_fov、priority。

### 4.10 控制算法模块

位置：

- `novasight/control/strategy.py`
- `novasight/control/output.py`
- `novasight/executors/runtime.py`

当前职责：

- 根据目标 aim 点计算 dx/dy。
- PID / proportional / predictive。
- 限幅、死区、平滑、指令合并。

边界评价：

- 策略接口较清晰。
- `ControlCommandCoalescer` 是必要模块。
- 但控制量单位需要更明确：px error、counts、kmNet dx/dy 分层。

已修复问题：

- 合并指令时保留 `move_kind`、`move_ms`、`trace_ms`、`bezier_ctrl`。

当前风险：

- 用户修改参数后，需要知道它影响的是哪一层：
  - `kp_x/kp_y` 影响比例项。
  - `ki/kd` 影响积分和微分。
  - `kp_x_move_max/kp_y_move_max` 是最终轴限幅。
  - `aim_ratio` 影响目标框内 aim 点。
  - `prediction_factor` 影响运动预测点。

建议：

- 前端参数设置页分组：
  - 目标选择
  - Aim 点
  - PID 参数
  - 输出限幅
  - kmNet 输出方式
  - 触发方式

### 4.11 执行器与 kmNet 模块

位置：

- `novasight/executors/runtime.py`
- `novasight/executors/kmnet.py`
- `novasight/executors/kmnet_loader.py`
- `novasight/api/routes_executors.py`

当前职责：

- 选择执行器。
- 限幅后输出。
- kmNet 连接、监听、移动、画圆诊断。

边界评价：

- executor registry 是合理 seam。
- kmNet executor 已暴露 `move_count`、`last_dx/dy`、`last_error`，可作为 UI 判断依据。

当前风险：

- `sent=True` 只代表 driver 调用成功，不代表物理感知一定成功。
- 连接态、监听态、输出态需要拆开。

建议：

- kmNet UI 必须同时展示：
  - 驱动可用
  - 已连接
  - 监听中
  - 最近发送
  - 发送次数
  - 最近错误
  - 算法执行器是否为 kmNet

### 4.12 授权许可模块

位置：

- `novasight/license.py`
- `novasight/api/app.py`
- `web/src/features/license`

当前职责：

- license 激活、校验、权限列表。
- API middleware 阻断未授权请求。

边界评价：

- 本地 license store 简洁。
- 测试卡密支持 bring-up。
- 但权限虽然存在，很多 API 只是统一要求 license，没有按 feature 细粒度判断。

当前风险：

- 用户界面需要体现授权带来的权益，而不是只显示“已授权”。
- created_at 不展示是对的，前端应展示当前时间、激活时间、到期时间、剩余时间。

建议：

- license UI 展示：
  - 当前套餐
  - 可用能力
  - 到期倒计时
  - Jetson 绑定信息
  - 测试授权标识

### 4.13 前端 API 模块

位置：

- `web/src/api.ts`

当前职责：

- 集中 API path。
- 处理 base URL / WebSocket URL。
- 请求 JSON。
- 类型定义。

边界评价：

- 集中 API 是正确方向。
- 但类型大量使用 `Record<string, unknown>`，前端不得不自行推断 runtime 结构。

当前风险：

- 后端返回状态结构变化时，前端不会立即编译失败。
- 配置应用结果不够结构化，UI 无法准确表达“回滚 / 生效 / 需要重启”。

建议：

- 为 runtime `vision/control/execution/inference` 建正式 TS 类型。
- Config update response 升级后同步 TS 类型。

### 4.14 前端主控台模块

位置：

- `web/src/features/studio/StudioConsoleView.tsx`
- `web/src/styles.css`

当前职责：

- 采集设置。
- 模型推理。
- 参数设置。
- 统计、延迟、kmNet 控制面板。

边界评价：

- 当前页面已经接近产品主控台。
- 但单文件承担太多逻辑，长期维护风险较高。

已修复问题：

- 配置失败后刷新真实后端状态。
- 本地触发绑定支持显式选择鼠标左键。
- kmNet move 反馈显示发送次数和最近 dx/dy。

当前风险：

- 用户主操作区和工程调试信息仍有混杂。
- 平板/Android 交互需要继续降低布局复杂度。

建议拆分：

```text
StudioConsoleView
  -> CaptureSettingsPanel
  -> InferenceSettingsPanel
  -> ControlParamsPanel
  -> KmNetPanel
  -> RuntimeMetricsPanel
  -> PreviewPanel
```

## 5. 本轮已完成修复

### 5.1 采集切换失败保护

问题：

- 用户切换 ROI 或采集配置时，新 GStreamer pipeline 打不开会导致旧 capture session 被停掉。

修复：

- `CaptureService.configure()` 记录 previous profile。
- 新配置失败时尽量恢复旧 profile。
- 无旧运行链路时才进入 unavailable 状态。

影响：

- 错误配置不会轻易打坏上一条可用采集链路。

### 5.2 采集源重建后 frame_id 回退

问题：

- frame source 重建后 frame_id 从 1 开始。
- 推理线程和预览流都可能继续等待大于旧 frame_id 的新帧，导致卡死。

修复：

- Runtime pipeline 检测 latest frame_id 小于等于 last consumed 时重置游标。
- MJPEG preview 做同样处理。

长期方案：

- 引入 `capture_generation`，彻底替代纯 frame_id 判断。

### 5.3 控制指令合并丢参数

问题：

- `ControlCommandCoalescer` 合并 dx/dy 后重新构造 `ControlIntent`，丢失 `move_kind/move_ms/trace_ms/bezier_ctrl`。

修复：

- 合并后的 intent 保留这些字段。

影响：

- 前端设置 kmNet 移动方式后，执行链路不会因为合并器丢配置。

### 5.4 前端配置失败回读

问题：

- 配置同步失败后，前端可能继续显示用户刚选但未生效的配置。

修复：

- Studio 和 Devices 配置失败后主动刷新 runtime。

影响：

- UI 更接近后端真实状态。

## 6. 当前最高优先级缺陷清单

### P0：配置应用事务化

症状：

- 配置更新分散在多个路由。
- ROI、capture、hardware、executor、inference 的应用顺序不统一。

建议：

- 实现 `RuntimeReconfigurator`。

### P0：frame identity 正式化

症状：

- 采集重建后 frame_id 回退需要补丁式处理。

建议：

- `CapturedFrame` 增加 `generation`。
- Runtime / preview 使用 `(generation, frame_id)`。

### P1：运行状态诊断模型

症状：

- 用户经常不知道为什么没有推理、为什么没有目标、为什么没有 move。

建议：

- 增加 `RuntimeFrameTrace`。
- 前端展示“当前卡在哪一层”。

### P1：模型运行状态闭环

症状：

- 模型仓库和推理运行态关联不够强。

建议：

- 模型页显示：
  - 当前运行 artifact
  - 准备切换 artifact
  - 切换是否成功
  - 推理后端状态

### P1：Studio 主控台拆分

症状：

- `StudioConsoleView.tsx` 单文件过大。

建议：

- 拆为多个 panel，每个 panel 只负责一个用户任务。

## 7. 建议实施路线

### 阶段 1：运行态稳定

目标：

- 不因配置切换破坏旧链路。
- 所有重建都有回滚。

任务：

1. 扩展 `RuntimeReconfigurator`，覆盖采集选择、模型切换和硬件连接变更。
2. 配置更新返回 `ConfigApplyReport`。
3. 引入 `capture_generation`。
4. 采集、ROI、模型、kmNet 热更新统一走 reconfigurator。

### 阶段 2：诊断闭环

目标：

- 用户知道系统为什么没有输出。

任务：

1. 增加 `RuntimeFrameTrace`。
2. 前端增加“链路状态条”：
   - 采集
   - ROI
   - 推理
   - 目标
   - 控制
   - 执行
3. 每一层显示 OK / blocked / failed / reason。

### 阶段 3：前端产品化

目标：

- 普通用户看到的是主操作，不是工程碎片。

任务：

1. 拆分 Studio panel。
2. 模型仓库改为“模型选择与运行状态”。
3. 工程调试信息统一放折叠区。
4. 平板/Android 页面切换体验优化。

### 阶段 4：真机可靠性

目标：

- Jetson 真机长时间运行稳定。

任务：

1. 采集设备拔插测试。
2. kmNet 断开 / 重连测试。
3. TensorRT engine 不兼容测试。
4. 运行 30 分钟稳定性观测。
5. FPS / latency / skipped / dropped 窗口统计验证。

## 8. 验收标准

一个健康版本至少满足：

- 修改 ROI size，成功则链路继续；失败则旧链路继续。
- 切换采集格式失败时，旧采集继续。
- 切换模型失败时，旧模型继续。
- kmNet 连接失败时，控制算法仍可 dry-run 诊断。
- 前端能准确说明没有输出的原因。
- 预览 FPS 慢不会影响推理 FPS。
- 所有关键配置变更都能看到是否已生效。

## 9. 结论

NovaSight 已经从原型进入产品化前夜。现在最需要的不是继续增加功能，而是把运行态治理做扎实：

- 统一配置应用。
- 统一运行诊断。
- 统一 frame identity。
- 统一模型运行状态。
- 统一前端产品表达。

这几个基础打稳后，再继续扩展模型管理、TensorRT 优化、kmNet 控制体验，才不会继续出现“修一个点又断另一条链路”的问题。
