# 算法责任与安全自检地图

本图随工作空间源码维护，不代表某个旧提交的运行收据。每次自检先记录 `git rev-parse HEAD`，再检查下方链接对应的当前实现。`YES` 只表示代码契约或指定测试通过；真实画面、Jetson 和物理输出单独验收。

```mermaid
flowchart LR
    I[硬件采集 / ROI / TensorRT] --> D[CUDA decode / 排序 / NMS]
    D -->|有效 DetectionBatch| T[TargetingCore<br/>准入、关联、选择]
    C[CrosshairHub<br/>当前参考点] --> T
    T -->|目标观测| A[AimAlgorithm<br/>预测、Atan、量化]
    A -->|OutputPlan| O[Device Worker<br/>门控、压枪、限幅]
    O --> K[kmNet 发送及回执]
    R[RuntimeSupervisor<br/>配置与 epoch] -.-> T
    R -.-> A
    R -.-> O
    F{{公平性风险评估<br/>尚未实现}} -.->|仅允许阻断 / dry-run| O
```

| 检查点 | 责任、当前证据和结论 | 验收级别 |
| --- | --- | --- |
| 感知到有效结果 | [GPU 结果校验](../crates/novasight-tensorrt/src/gpu.rs)只接受有界最终结果；[正式构造](../apps/novasightd/src/live_perception.rs)选择 TensorRT。`PARTIAL`：支持的模型契约已实现，真实非空检测与全格式通用性未通过。 | 旧 Jetson 收据；当前 SHA 待验 |
| 准星超时 | [CrosshairHub](../crates/novasight-pipeline/src/crosshair.rs)的 `observation_stale` 必须回退几何中心，不得继续持有过期视觉点。已补回归检查；当前工作区验证待统一执行。 | 源码 + 主机测试待验 |
| 多目标与不同 cls | [关联](../crates/novasight-core/src/tracking/association.rs)把 raw cls 变化作为软代价；[选择](../crates/novasight-core/src/tracking/selection.rs)综合距离、大小、类别、置信度、连续性和运动；[拥挤帧](../crates/novasight-core/src/tracking/mod.rs)先按业务分数排序再限 16 个并保留当前锁定。`PARTIAL`：域测试不等于真实高速遮挡手感。 | 39 项主机合同；Jetson 待验 |
| 目标丢失 | [TargetingCore](../crates/novasight-core/src/tracking/mod.rs)保留短期身份，但丢失目标不发控制目标；不要把“等待身份”画成继续移动。 | 主机合同；现场待验 |
| 预测与控制 | [AimAlgorithm](../crates/novasight-core/src/controller/algorithm.rs)只处理当前观测的预测、Atan 和量化；[交付 worker](../crates/novasight-pipeline/src/runtime.rs)重新检查 generation / gate 并在压枪后限幅。`PARTIAL`：自身运动、检测噪声和动作延迟未由真实数据标定。 | 源码 + 主机合同；设备待验 |
| 瞄点热更新 | [目标配置更新](../crates/novasight-core/src/tracking/mod.rs)在瞄点比例变化时重建身份和预测历史；普通评分调参不重建。已补回归检查，当前工作区验证待统一执行。 | 源码 + 主机测试待验 |
| 状态有效性传递 | [TargetingCore](../crates/novasight-core/src/tracking/mod.rs)的 Kalman 只参与身份关联；控制用当前 raw aim 和自己的短窗预测。未被 [交付](../crates/novasight-pipeline/src/runtime.rs)消费的 `target_state_valid` 已删除，避免把 Kalman 可靠性误称为控制预测开关。 | 源码 + 主机测试待验 |
| 物理输出 | [Device Worker](../crates/novasight-pipeline/src/runtime.rs)是唯一发送点；触发、当前代、输出门与回执须在设备侧证明，不能用 UI“已生效”代替回执。 | 主机合同；实机未知 |
| 公平性异常 | “光标持续粘附目标”属于外部规则/产品合规风险，不能由检测准确率证明无异常。`NOT IMPLEMENTED`：若评估成立，只能阻断输出并记录 dry-run；不得引入伪装人类操作或规避检测的运动生成。 | 需求与安全边界；待实现/验收 |

自检顺序：先核实输入和结果有效，再核实目标选择与缺失期间零旧命令，再核实预测/控制与输出回执；出现异常时保留同一帧的 `epoch / generation / captured_at`、目标 ID、门控决定和原始错误。禁止拿旧提交收据、空检测或模拟前端页面证明当前物理行为。

历史交互地图 `2026-08-31/develop-alpha@350a38f` 仅是当时的讨论快照；其中 `tracking.rs` 等路径与“已确认冲突”标签不能沿用为当前工作区事实。
