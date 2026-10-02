# NovaSight 当前主线 TODO

本文只描述当前 Rust 产品主线。旧 Python Angular PD、轨迹 Scheduler、
动态限幅、arrival radius、residual cap 和多执行器方案不再是实现依据。

## 唯一控制链

```text
DetectionBatch latest-only
-> 目标选择与身份跟踪
-> 4 点短窗 / 3 段 medoid 速度预测
-> frame age + actuation delay + lead
-> 连续 Atan 控制
-> 整数 counts 量化
-> 固定 X/Y 设备限幅
-> generation / trigger / output gate 复核
-> kmNet move(dx, dy)
```

主链不允许第二套控制公式、动态输出上限、轨迹拆分或未声明的 counts
补偿器介入。目标置信度只决定目标是否进入选择，不参与控制增益。

## 当前开发顺序

1. 保持参数页面与上述链路同序：触发延迟、预测/算法和固定限幅；
   运行总开关只在首页出现。
2. 普通页面只保留能够解释实际产品行为的参数；Tracker/Kalman 标定项
   只能进入专家区域，回放实现细节不得成为产品参数。
3. 配置写入必须同时持久化并热更新；只有真正的进程级基础项允许提示
   下次启动接管。
4. UI 状态必须来自后端单一语义状态和单调 revision，不允许多个异步
   布尔值分别推导同一个界面结论。
5. 默认构建不编译评估、回放分析和历史诊断模块；需要时通过显式 feature
   开启。

## Jetson 待验收

- 无模型时 daemon 和 Studio 能正常启动，模型主链保持等待态。
- 用户可见 Studio 固定监听 `0.0.0.0:7351`；源码开发时私有
  Web/API 仅监听 `127.0.0.1:5174`，局域网浏览器使用启动器打印的认证 URL。
- 配置保存后 YAML revision、运行时 effective revision 和页面状态一致。
- 真实 DetectionBatch 时间戳与 freshness 门控使用同一 monotonic 时钟域。
- 预测关闭、预测开启、目标切换和短暂丢失场景没有旧命令补发。
- 最终设备输出严格满足：

  ```text
  out_x = clamp(tracking_x, -max_x, max_x)
  out_y = clamp(tracking_y, -max_y, max_y)
  ```

- kmNet 硬件触发释放后不存在漏发；最新 generation 以外的命令不会到达
  设备。
- 记录真实静态目标、横向移动、纵向移动和遮挡，再校准 FOV、
  counts/360、响应参数与执行延迟。

## 完成边界

- 开发机静态检查不能替代 Jetson、采集卡、TensorRT 和 kmNet 实机证明。
- 当前本地优化批次在实机验收前不宣称端到端完成。
- 验收通过后再提交并推送，提交状态与推送状态分别报告。

## 消费级 Studio 后续外观

### 白红 · Pulse White 完整外观

**What:** 在消费级 Studio 核心结构稳定后，补齐 Pulse White 的完整令牌、组件状态、主题预览、持久化与跨标签页同步。

**Why:** 用户明确喜欢白红视觉，但第一版先用 Graphite Signal 证明五空间、Eye / Body / Nerve 和旧 CSS 删除能够完整落地，避免主题矩阵拖慢结构替换。

**Context:** 2026-09-23 CEO 评审 D2/R1 选择延期。实现时不得只换颜色：需要覆盖 loading、empty、applying、effective、restart-required、warning、error、disabled、focus 和 reduced-motion；状态意义必须同时依靠文字与图标；外观选择器只有在完整可用时才显示该选项。相关设计合同位于 `docs/designs/novasight-continuous-living-instrument.md`。

**Effort:** human S / CC+gstack S
**Priority:** P2
**Depends on:** Graphite Signal 下五个产品空间完成替换，旧全局样式已删除或完成归属迁移

### 拉花竞技 · Track Livery 完整外观

**What:** 在最终页面几何稳定后，实现 Track Livery 的竞技斜切、号码牌、速度带和非对称外壳构图。

**Why:** 该外观是用户要求的独特汽车竞技皮肤，但它依赖稳定的导航、页面边缘和内容区域；第一版先删除旧结构，避免自定义图形跟随布局反复重做。

**Context:** 2026-09-23 CEO 评审 D3/R2 选择延期。实现必须遵守 12–18° 斜切、可见外壳装饰不超过 12–15%、不得跨越实时视频、参数、日志、图表或错误文字；同时覆盖 768/1024/1280/宽屏、键盘焦点、对比度和 reduced-motion。只有完成这些验收后才能加入外观选择器。

**Effort:** human M / CC+gstack S
**Priority:** P2
**Depends on:** 五个产品空间与最终 Shell 几何稳定，Graphite Signal 验收完成

## 消费级 Studio 后续能力

### Nerve 时间轴历史缩略图

**What:** 在现有 60 秒事件与 2 Hz 指标时间轴上，增加仅随可见预览消费者采样的 1 Hz 历史缩略图和事件留图。

**Why:** 让用户能够沿时间轴回看“当时看到了什么”，把实时画面、参数动作、指标和日志组成完整视觉证据链。

**Context:** 2026-09-23 CEO 评审 D4/R3 选择延期，决策 `f11e180d-7e9e-461e-99b8-ec87d85f26a9`。第一版完整度上限为 7/10：保留实时 Eye、60 秒结构化事件和 2 Hz 指标，但不能回看历史画面。升级条件是五空间 Shell 与事件/指标时间轴稳定，并在 Jetson 输出关闭条件下证明 active-preview-only JPEG 留存不会降低推理主链性能。实现必须复用 `PreviewHub` 已编码的 `Arc<[u8]>`，不得新增 CPU 解码/重编码或无人查看时的后台编码；需要年龄/数量/字节上限、时间身份、最近样本容差和明确空缺。后续实现该简化边界时，在时间轴图像接缝标注 `gstack-shortcut(dec-f11e180d): no retained visual evidence in first release, upgrade after shell stability and Jetson profiling`。

**Effort:** human M / CC+gstack S plus Jetson profiling
**Priority:** P2
**Depends on:** 五空间 Shell、结构化活动环与 60 秒指标时间轴稳定

### 待重启参数的一键撤销

**What:** 为“已保存但尚未生效”的参数补充单项撤销和全部撤销操作，并在完成后重新读取后端 canonical 配置。

**Why:** 第一版已能显示 desired/effective 差异并通过安全重启完成生效，但撤销多个待重启字段仍需逐项改回运行值；专用操作可降低批量修正成本。

**Context:** 2026-09-23 CEO 评审 D5/R4 选择延期，第一版完整度为 8/10。实现时必须复用 schema-driven 参数写入路径，不得建立第二套配置状态；需要覆盖 revision 冲突、批量部分失败、canonical reread、按钮禁用/忙碌状态、键盘焦点和错误恢复。无论是否实现该便利层，待重启字段清单、安全 output-off 重启、预期断连/重连和重启后 effective 核对都属于现有必需能力。

**Effort:** human S / CC+gstack S
**Priority:** P2
**Depends on:** schema-driven apply 路径及 restart-required 状态在真实 Jetson 流程中稳定
