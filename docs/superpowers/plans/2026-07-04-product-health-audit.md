# NovaSight 产品健康审计

日期：2026-07-04

## 审计目标

这次不是单点 bug 修复，而是按真实产品运行链路重新审视主要模块：

采集输入 -> ROI 帧资源 -> 推理运行时 -> 目标选择 -> 控制量计算 -> kmNet 执行 -> 前端状态反馈。

核心原则：

- 新配置失败不能破坏上一组可用运行链路。
- 前端展示必须来自后端真实运行态，不能靠猜测。
- 可热更新的参数要实时生效；需要重建模块的参数必须有 prepare/apply/rollback。
- 采集、推理、控制必须能解释当前为什么工作或为什么不工作。

## 主要模块边界

### 配置模块

位置：`novasight/config/runtime.py`、`novasight/api/routes_runtime.py`

职责：

- 严格解析和校验运行配置。
- 将配置应用到 capture/runtime/inference/executor/hardware。
- 保存配置文件。

当前风险：

- `PUT /api/config` 仍然承担太多 orchestration 责任。
- `active_config` 和 `desired_config` 尚未正式分离。
- 运行模块热更新结果还没有结构化返回给前端。

产品化方向：

- 新增 `RuntimeReconfigurator`，统一处理 prepare/apply/rollback。
- 返回 `applied_sections`、`rollback`、`active_config`、`runtime_impact`。

### 采集模块

位置：`novasight/capture/service.py`、`session.py`、`source.py`、`pipeline.py`

职责：

- 查询采集能力。
- 选择 GStreamer appsink pipeline。
- 维护单路最新帧采集会话。

已修复问题：

- 采集配置切换失败时不再直接破坏上一条可用采集链路。
- 新采集管线打开失败后尽量恢复旧 profile。

剩余风险：

- 真机设备独占时，部分切换仍需要先释放旧 pipeline；如果旧 pipeline 恢复失败，需要更明确的 UI 状态。
- GStreamer candidate 尝试策略仍偏底层，需要将“推荐链路”和“诊断链路”分开表达。

### 预览模块

位置：`novasight/api/routes_capture.py`、`novasight/capture/preview.py`

职责：

- 限速 MJPEG 输出。
- 使用后端真实 frame 和 runtime detection 绘制预览。

已修复问题：

- 采集源重建后 `frame_id` 从 1 开始，预览流不再永久等待旧 frame_id。

剩余风险：

- 预览是消费者，不应影响主线推理；后续需要更明确地显示 preview fps 与 capture/inference fps 的隔离关系。

### 推理模块

位置：`novasight/inference/runtime.py`、`onnxruntime_engine.py`、`tensorrt.py`

职责：

- 根据产物后缀自动选择 ONNX Runtime 或 TensorRT。
- 安全加载、切换模型。
- 输出标准检测结果和 debug 信息。

当前风险：

- 模型输入 shape、ROI size、推理 resize 的关系需要前端用产品语言解释。
- TensorRT 失败信息已经能返回，但还需要更好的用户分级提示：环境缺失、engine 不兼容、模型输出无法解码。

产品化方向：

- 模型切换返回 `prepare_status` 和 `commit_status`。
- 模型仓库显示“当前运行模型”和“可切换模型”，避免用户误以为选择了就已经运行。

### 运行流水线模块

位置：`novasight/runtime/pipeline.py`、`runtime/service.py`

职责：

- 消费采集最新帧。
- 串联 ROI、推理、目标选择、控制、执行。

已修复问题：

- 采集重建后 frame_id 回退时，推理线程会重置消费游标，不再卡死。

剩余风险：

- pipeline 还没有显式的 generation/session_id。长期应使用 `capture_generation + frame_id`，而不是只靠 frame_id 判断。

### 控制模块

位置：`novasight/control/strategy.py`、`novasight/executors/runtime.py`

职责：

- 目标选择后计算 dx/dy。
- 限幅、合并、输出到执行器。

已修复问题：

- 指令合并器保留 `move_kind`、`move_ms`、`trace_ms`、`bezier_ctrl`，避免前端配置的 kmNet 移动方式在合并后丢失。

剩余风险：

- 控制量坐标系需要继续保持透明：ROI 坐标、模型输入坐标、aim 点、raw dx/dy、counts dx/dy 应分层展示。
- 触发方式、按键绑定、硬件回传触发应统一成一个 `TriggerState`。

### kmNet 执行模块

位置：`novasight/executors/kmnet.py`、`kmnet_loader.py`

职责：

- 加载本地 kmNet driver。
- 连接、监听、发送 move/left/trace。

当前风险：

- driver 成功返回不等于物理鼠标一定移动，需要 UI 展示 `sent`、`move_count`、`last_dx/dy` 和 `last_error`。
- 连接态、监听态、输出态应被前端拆开，不应只显示“已连接”。

### 前端主控台

位置：`web/src/features/studio/StudioConsoleView.tsx`

职责：

- 作为普通用户主操作区。
- 配置采集、模型、参数、kmNet。
- 展示真实运行反馈。

已修复问题：

- 配置同步失败后主动刷新后端真实状态，避免 UI 显示未生效配置。

剩余风险：

- 页面仍承担过多职责，需要拆分成更深的前端模块：采集设置、模型设置、参数设置、运行反馈、kmNet 控制面板。
- 前端 API 返回结果还不够结构化，导致很多状态只能从 runtime snapshot 推断。

## 这次代码修复清单

- `CaptureService.configure()` 失败保留或恢复旧采集链路。
- MJPEG 预览在采集源重建后处理 frame_id 回退。
- 控制指令合并保留 kmNet 移动参数。
- Studio / Devices 配置失败后刷新后端真实状态。

## 下一阶段建议

1. 建立 `RuntimeReconfigurator`，统一所有运行中配置变更。
2. 引入 `capture_generation`，彻底解决采集重建后的 frame identity 问题。
3. 将 `PUT /api/config` 返回结构升级为配置应用报告。
4. 前端将“当前运行配置”和“正在编辑配置”分开。
5. kmNet 增加物理移动诊断状态流，区分 driver sent 和用户可感知移动。
6. 推理页面增加模型运行状态闭环：已选择、准备中、已提交、执行失败、当前运行。

