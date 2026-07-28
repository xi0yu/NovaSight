# 视觉准星控制基准

NovaSight 可以从主链 ROI 中学习固定 HUD 准星，并将经过连续确认的准星中心作为目标选择和鼠标控制的共同参考点。该功能默认关闭；未学习、未确认、观测过期或分辨率变化时，系统自动使用原有几何中心。

## 数据通路

```text
DeepStream ROI NVMM tee
├─ nvinfer / DetectionBatch / Tracker / Control
└─ leaky latest-only queue
   → videorate
   → nvvidconv 中心 96×96
   → nvjpegenc
   → CrosshairSystem
```

准星支路不读取 `LatestFrameBroker`，不会消费或延迟推理帧。它只保留最新样本，默认以 10 Hz 运行。若 Jetson 缺少 JPEG 支路所需插件或发生 caps 协商失败，DeepStream 会关闭可选 JPEG 支路并重试主推理链。

## 学习和确认

1. 在参数设置中启用“低频准星观测”，然后启动或重启主链。
2. 保持正常准星可见，等待学习帧数量满足要求。
3. 点击“学习当前准星”或在参数页面按 F8。
4. 系统使用最近多帧的中位图提取中心附近的稀疏颜色与灰度结构模板。
5. 后续观测先在完整允许范围内粗搜，再在最佳位置附近精搜；稳定运行时只搜索上一确认位置附近。
6. 连续匹配达到确认时间后，才允许视觉中心接管控制基准。

模板保存在 `data/crosshair/template.json`，模板预览由 `/api/crosshair/template.png` 提供。清除模板会立即恢复几何中心。

## 控制契约

控制侧只读取 `ControlReference`：

```text
x / y
source: geometry | vision_verified | vision_hold
confidence
sample_ts_ns / age_ms
geometry_signature
reason
```

同一个 `generation + frame_id` 只解析一次参考点，所以目标候选过滤、V2 控制和其他鼠标算法不会在同一帧使用不同中心。短时匹配失败可以在 `max_age_ms` 内保持最后一次已确认中心；超过时效立即回退几何中心。

## 配置边界

日常使用只需要：

- `enabled`：建立独立采样支路；变更需要重启主链。
- `use_for_control`：允许已确认视觉中心接管控制。
- `search_size`：中心采样尺寸，默认 96。
- `sample_hz`：低频观测频率，默认 10 Hz。
- `sample_frames`：学习使用的最近帧数，默认 5。

确认时长、最大时效、匹配阈值、最大中心偏移和单次平滑步长保留为配置文件中的工程参数，不在日常界面展开。

## 当前范围

当前版本针对固定在屏幕中心附近的 HUD 准星。它不估计后坐力，不替代目标 Tracker，也不把准星和压枪分别发送到设备。间隔式压枪只在满足时把 `+Y` 合入当前控制命令，最终继续走统一的 Latest Replace 交付路径。
