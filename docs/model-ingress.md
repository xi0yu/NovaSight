# NovaSight TensorRT 模型接入

NovaSight 正式控制主线只启用经过 `ModelProfile` 诊断验证的 TensorRT Engine。模型目录或注册表中的 `input_shape`、ROI 尺寸和文件名不再作为 Engine 输入契约来源。

正式接入顺序：

```text
POST /api/models/artifacts/{id}/inspect
→ PUT /api/models/artifacts/{id}/profile
→ POST /api/models/artifacts/{id}/probe
→ POST /api/models/projects/{project_id}/publish
```

Studio HTTP 接口和 `novasightctl model` 子命令都只连接 `novasightd`，并通过同一个 typed `RuntimeCommand` 进入 `RuntimeSupervisor`；它们不直接启动 Python、写 manifest 或修改 SQLite。对应 CLI 为：

```text
novasightctl model inspect <artifact-id>
novasightctl model profile <artifact-id>
novasightctl model configure <artifact-id> --request <profile.json>
novasightctl model probe <artifact-id> --input-mode fixed
```

模型接入由 Rust 直接执行。Cargo 在 Jetson 构建时编译并静态链接窄 TensorRT/CUDA C++ ABI；Rust 负责 Engine 身份、语义配置、decoder/NMS/DetectionBatch 验证、manifest 原子写入与 catalog 提交。API 请求不能指定动态库、脚本、命令或工作目录，也不会启动 Python worker。

`inspect` 使用 TensorRT API 反序列化 Engine，并读取真实 I/O Tensor 名称、解析后的运行 Shape 与 DataType。静态 Engine 直接采用声明 Shape；动态 Engine 明确采用 Optimization Profile 0 的 OPT Shape，并保留动态标记和 profile index。当前窄 ABI 未读取的 Tensor Format、Shape Tensor 状态、profile min/max 与向量化元数据返回 unknown/null，不会用默认值伪装成探测结果。首期只接受单个 NCHW、Batch 1、三通道图像输入，以及已登记的检测输出解析器。

Engine 不能可靠保存的 RGB/BGR、归一化、resize/letterbox、类别、bbox 与 objectness 语义必须在 `profile` 阶段确认。接入状态、验证报告和 DeepStream 运行契约统一保存在与 Engine 同目录的 `<engine>.manifest.json`；Engine 的 SHA-256 或文件大小变化会使验证立即失效。

`probe` 使用独立候选 TensorRT Context，强制取消控制输出并暂停正在运行的推理主链。请求体 `{"input_mode":"fixed"}` 已实现固定零输入的真实执行契约检查。诊断会检查 Engine 执行、输出 NaN/Inf、Decoder、NMS、bbox/class 合法性以及 `DetectionBatch` 构造；候选 Context 在诊断结束后关闭，不替换当前活动 Engine，原 Runtime 随后恢复。`{"input_mode":"latest"}` 必须等 Rust capture 的 latest-frame 单槽进入 RuntimeSupervisor；当前明确返回 `409 MODEL_LATEST_FRAME_UNAVAILABLE`，不会用固定输入伪装成功。

验证通过后，Rust 把标准 runtime manifest 与 WebUI 兼容 profile 原子合并到同一个 `<engine>.manifest.json`。随后重新计算 Engine SHA-256，并在 SQLite immediate transaction 中同步 artifact 状态、类别和输入 Shape；验证或数据库提交失败时恢复操作前的 manifest。活动 artifact 以及与活动 artifact 共享 version 元数据的 sibling artifact 都禁止重新 inspection、configure 或 probe。正式发布会再次校验 Engine 内容标识和完整 fixed-probe receipt；没有四项通过标志及 manifest fingerprint 的 Engine 无法进入控制链。运行时只临时重建 `data/runtime/deepstream/active-nvinfer.ini`。

模型目录浏览和强制重新校验都是只读操作。选择未登记 Engine 时，Registry 只保存原始文件路径和内容标识，不复制 Engine，也不创建哈希版本资产目录。

旧部署若仍只有同目录通用 `model.manifest.json`，兼容启动会将它原子迁移为对应的 `<engine>.manifest.json`，成功后删除旧文件；不会再生成新的通用 manifest。显式离线导入/构建工具不属于模型选择流程，只有人工调用时才会创建转换产物。

模型切换采用停止重启：停止控制与推理、清除 latest frame 和控制状态、提交新 Engine、重建运行管线。旧帧和旧目标状态不会跨模型继续执行。

当前 Jetson 自定义 NVMM CUDA 预处理只精确支持 RGB、`1/255` 和直接缩放；不符合该契约的 ModelProfile 会明确失败。DeepStream 能表达的其他预处理语义由生成配置负责，不能静默退回默认值。
