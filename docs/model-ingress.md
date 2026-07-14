# NovaSight TensorRT 模型接入

NovaSight 正式控制主线只启用经过 `ModelProfile` 诊断验证的 TensorRT Engine。模型目录或注册表中的 `input_shape`、ROI 尺寸和文件名不再作为 Engine 输入契约来源。

正式接入顺序：

```text
POST /api/models/artifacts/{id}/inspect
→ PUT /api/models/artifacts/{id}/profile
→ POST /api/models/artifacts/{id}/probe
→ POST /api/models/projects/{project_id}/publish
```

`inspect` 使用 TensorRT API 反序列化 Engine，并读取 I/O Tensor 名称、模式、Shape、DataType、Tensor Format、Shape Tensor 状态和 Optimization Profile。首期只接受单个 NCHW、Batch 1、三通道图像输入，以及已登记的检测输出解析器。

Engine 不能可靠保存的 RGB/BGR、归一化、resize/letterbox、类别、bbox 与 objectness 语义必须在 `profile` 阶段确认。确认结果保存在与 Engine 同目录的 `<engine>.profile.json`，其 SHA-256 与文件大小变化会使验证立即失效。

`probe` 使用独立候选 TensorRT Context，强制取消控制输出并暂停正在运行的推理主链。请求体 `{"input_mode":"fixed"}` 使用固定零输入检查执行契约，`{"input_mode":"latest"}` 在控制隔离后读取 capture 单槽中的最新 ROI 帧，不建立诊断帧队列。诊断会检查 Engine 执行、输出 NaN/Inf、Decoder、NMS、bbox/class 合法性以及 `DetectionBatch` 构造。候选 Context 在诊断结束后关闭，不替换当前活动 Engine。

验证通过后，系统从同一个 `ModelProfile` 生成兼容的 `model.manifest.json` 与 DeepStream nvinfer 配置。正式发布会再次校验 Engine 内容标识；没有完整验证报告的 Engine 无法进入控制链。

模型切换采用停止重启：停止控制与推理、清除 latest frame 和控制状态、提交新 Engine、重建运行管线。旧帧和旧目标状态不会跨模型继续执行。

当前 Jetson 自定义 NVMM CUDA 预处理只精确支持 RGB、`1/255` 和直接缩放；不符合该契约的 ModelProfile 会明确失败。DeepStream 能表达的其他预处理语义由生成配置负责，不能静默退回默认值。
