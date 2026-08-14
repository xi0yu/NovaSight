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
-> 压枪 +Y 合成
-> 固定 X/Y 设备限幅
-> generation / trigger / output gate 复核
-> kmNet move(dx, dy)
```

主链不允许第二套控制公式、动态输出上限、轨迹拆分或未声明的 counts
补偿器介入。目标置信度只决定目标是否进入选择，不参与控制增益。

## 当前开发顺序

1. 保持参数页面与上述链路同序：触发方式、开火延迟、预测/算法、压枪、
   固定限幅、输出。
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
- 固定监听地址为 `0.0.0.0:5174`，局域网浏览器能够访问 Studio。
- 配置保存后 YAML revision、运行时 effective revision 和页面状态一致。
- 真实 DetectionBatch 时间戳与 freshness 门控使用同一 monotonic 时钟域。
- 预测关闭、预测开启、目标切换和短暂丢失场景没有旧命令补发。
- 最终设备输出严格满足：

  ```text
  out_x = clamp(tracking_x, -max_x, max_x)
  out_y = clamp(tracking_y + recoil_y, -max_y, max_y)
  ```

- kmNet 硬件触发释放后不存在漏发；最新 generation 以外的命令不会到达
  设备。
- 记录真实静态目标、横向移动、纵向移动、遮挡和压枪手感，再校准 FOV、
  counts/360、响应参数与执行延迟。

## 完成边界

- 开发机静态检查不能替代 Jetson、采集卡、TensorRT 和 kmNet 实机证明。
- 当前本地优化批次在实机验收前不宣称端到端完成。
- 验收通过后再提交并推送，提交状态与推送状态分别报告。
