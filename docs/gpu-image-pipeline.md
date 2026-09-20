# 图片到检测结果：当前 GPU 流水线契约

核对日期：2026-09-20。本文是图片处理与 YOLO 后处理的当前入口；旧设计、带日期的测量记录不能覆盖当前代码。范围在有效 `DetectionBatch` 交付处结束，跟踪、目标选择、预测、控制和设备输出不属于本次优化。

## 执行责任

| 阶段 | 当前执行者 | 代码位置 |
| --- | --- | --- |
| V4L2 输入；MJPEG 像素解码 | NVIDIA 硬件解码器；原始 NV12/YUY2 不需要压缩解码 | [pipeline.rs](../crates/novasight-platform-jetson/src/deepstream/pipeline.rs) |
| ROI 裁剪、缩放、转 RGBA/NVMM | NVIDIA VIC，经 `nvvidconv` | 同上 |
| RGBA → RGB/BGR、NCHW、FP16/FP32、归一化 | CUDA | [rgba_preprocess.cu](../native/yolo-postprocess/src/rgba_preprocess.cu) |
| 模型推理 | TensorRT 设备输入／输出 API | [jetson.cpp](../native/tensorrt-runtime/src/jetson.cpp) |
| YOLO decode、过滤、稳定排序、完整类别感知 NMS、打包 | CUDA；排序使用 CUB | [yolo_gpu.cu](../native/yolo-postprocess/src/yolo_gpu.cu) |
| 最终结果交付 | 仅固定 6152 字节结果回传；Rust 验证有限元数据、绑定帧标识 | [gpu.rs](../crates/novasight-tensorrt/src/gpu.rs) |

这里的“全 GPU 图片路径”准确含义是：**像素处理、推理和候选后处理全部由 GPU／适用的 NVIDIA 专用硬件执行**。VIC 和硬件解码器不是 CUDA。CPU 仍负责媒体头解析、调度、资源管理和最终元数据校验；不在 CPU 上裁剪图片、构造模型张量、解码原始 YOLO 输出或重新执行 NMS。

原始采集格式可能从系统缓冲进入硬件表面，不声称所有输入都无需初始传输；NVMM 图片不得回传 CPU 处理后重新上传。当前生产入口是 V4L2 帧，不等于任意 JPEG/PNG 文件输入接口都已实现。

正式构造在 [live_perception.rs](../apps/novasightd/src/live_perception.rs) 选择 `InferenceStage::TensorRt` 和 GPU 配置；不会选择 `nvinfer` CPU parser。旧 DeepStream 构造、C++ parser 和 Rust decoder 只作为隔离对照／模型诊断保留，不是自动回退。遇到不支持的模型、布局或 GPU 错误，明确失败。

## 模型边界，不冒充通用支持

- Batch 1，三通道 NCHW，RGB/BGR，FP32/FP16 输入输出；直接缩放，不支持当前路径中的 letterbox。
- 单路 raw YOLO 输出 `[C,N]` / `[N,C]`，或带 Batch 1 维度；像素 `cxcywh`，已激活类别分数，可选 objectness，最佳单类别选择。
- 最多 32768 个候选、1024 类、16384 像素边长；每类 NMS 后 Top-K 上限 256。最终容量仍为 256，按类别顺序保留前缀并报告截断量，**不是全局分数最高的 256 个**。
- EfficientNMS、多输出 Rockchip head、未激活分数、INT8 I/O 等不受当前 GPU 准入支持，不猜测格式或换用 CPU。
- 归一化倍率必须为正，且 `255 × scale` 在输入 dtype 中可表示，防止 FP16/FP32 预处理溢出。

约束由 [model_contract.rs](../crates/novasight-platform-jetson/src/deepstream/model_contract.rs) 和 native 实际 engine binding 检查共同执行。这里不新增另一套 parser 或模型加载器。

## 有效结果的含义

1. 输入必须是符合模型尺寸、RGBA、单帧 NVMM/EGL 布局的表面；GPU 工作完成前保留其生命周期。
2. CUDA 完成后才交付结果；不能把 enqueue 成功当作结果成功。失败包含阶段和原始 SDK 原因，异步等待错误不被误标为确定的 NMS 根因。
3. 结果数量不超过 ABI 容量；数量加截断量不超过候选数及每类 Top-K 总上限；出现截断时结果必须已填满容量。
4. 每个框必须有限、尺寸为正且完整位于声明的模型坐标空间；类别合法，分数有限、在 `(0,1]` 内且达到配置阈值。只校验最多 256 个最终框，不回传 raw tensor 或在 CPU 重做 NMS。
5. 结果保留原 `epoch / generation / captured_at`；会话入口及交付处沿用已有新鲜度检查。空检测允许存在，但不等于识别质量通过。

有效数值不保证模型识别准确，也不证明 HDMI 信号正常。CUDA 对非法候选已有过滤，但尚无按原因细分的过滤计数；全部候选无效可能表现为空检测。这仍是可观测性缺口，不能用空结果宣称原始模型输出全部正常。

## 验证与尚未通过的验收

主机检查：

```sh
cargo test -p novasight-tensorrt --features ffi --locked
cargo test -p novasight-platform-jetson --features deepstream --lib deepstream::model_contract::tests --locked
cargo test -p novasight-platform-jetson --features deepstream --test deepstream_pipeline --locked
```

这些只证明主机契约／参考 ABI，不证明 CUDA 执行。Jetson 需按 [CUDA 模块说明](../native/yolo-postprocess/README.md) 构建并运行 GPU 差分测试、预处理检查和 Compute Sanitizer，再用已核实内容的非空真实图片／视频验证：

- 相同模型、输入和阈值下，类别、分数、框、排序、NMS 和截断结果符合参考语义。
- 追踪显示预处理、推理、decode、排序和 NMS 均走指定硬件，只有最终有界结果回传。
- 错误、切换与复用不会交付旧帧或无效结果；不测试或修改结果后的业务算法。
- 延迟从真实输入到最终结果测量；分开报告模型推理耗时，不用零输入或无信号画面证明识别性能。

[2026-09-08 接入收据](gpu-product-integration-20260908.md) 是历史 GPU 执行证据，不是本次代码改动的设备验收。正常非空场景、全部模型／格式覆盖、长期稳定性仍需目标机验证。没有 CUDA/Jetson 的开发机不得给这些项目标记“通过”。

## 文档责任

- 当前实现与范围：本文；模型登记：[model-ingress.md](model-ingress.md)。
- 历史测量：[GPU 基线](pipeline-gpu-baseline-20260908.md)、[接入收据](gpu-product-integration-20260908.md)。
- 旧 CPU／DeepStream 主线设计：`docs/archive/vision-legacy/`。原路径保留短入口避免断链，旧内容不再作为当前实施要求。
- [旧性能计划](pipeline-gpu-optimization-plan.md) 中更广的业务／设备控制验收不扩大本次任务；本次止于图片到有效检测结果。
