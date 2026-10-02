# 旧 DeepStream Object-Meta 主线（已归档）

当前正式路径见 [图片到检测结果：GPU 流水线契约](gpu-image-pipeline.md)。

旧方案的 `nvinfer → C++ parser → DeepStream CPU NMS` 不再是正式实时主线，也不是 GPU 失败后的回退路径。

原文保留在 [历史设计](archive/vision-legacy/novasight-deepstream-object-mainline.md)，只用于追溯，不能用其中的配置或验收项覆盖当前实现。
