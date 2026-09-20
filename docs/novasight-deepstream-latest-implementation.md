# 旧 latest-only 实现基线（已归档）

当前正式路径见 [图片到检测结果：GPU 流水线契约](gpu-image-pipeline.md)。正式入口采用 NVMM、TensorRT 设备张量和 CUDA decode／NMS，不使用 CPU 图像预处理主线。

旧的临时主线、实验后端和迁移步骤保留在 [历史基线](archive/vision-legacy/novasight-deepstream-latest-implementation.md)，不代表当前默认行为。
