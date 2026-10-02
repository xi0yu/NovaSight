# NovaSight 目标决策策略替换计划

状态：IMPLEMENTED

日期：2026-09-21
分支：`develop-alpha`
固定起点：`9f0a418`

## 目标

把当前“Track ID 优先”的跟踪与选择实现替换为“当前最佳目标优先”的目标决策模块，同时保持生产调用接口 `TargetingCore::select_at` 不变。

成功标准：

1. `Detection` 继续只表达模型提供的 `object_id / raw class_id / bbox / confidence`，不混入内部类别语义。
2. 内部类别策略、短时关联、目标评分与卡尔曼状态分别归入独立 Rust 模块。
3. 相同 `cls` 优先关联；跨 `cls` 可在空间、时间和尺度连续时降权关联，不再硬断轨。
4. 旧目标消失时：没有可靠替代目标才等待；存在持续可靠候选时按切换延迟接管，不被旧 ID 的完整丢失宽限期阻塞。
5. 目标评分同时消费类别、预测后距离、检测置信度、框尺度、轨迹连续性和运动趋势。
6. 前端按“类别策略 / 目标决策 / 短时关联”分组展示并编辑后端真实字段；保存仍走现有草稿、revision 和热更新链。
7. 旧配置自动迁移，新代码不保留第二套旧评分路径。

## 模块与 seam

```text
perception::Detection                 官方/模型原始事实
        │
        ▼
tracking::classification             内部 cls 偏好与跨类代价
        │
        ▼
tracking::association                Track ID、几何与短时运动关联
        │
        ▼
tracking::selection                  当前目标评分、保持、等待、切换
        │
        ▼
TargetingCore::select_at             唯一生产接口
```

- 外部 seam 不新增 adapter；这是纯内存深模块。
- 前端类别名称与人物瞄点角色仍是产品配置；运行时只消费压平后的 active policy。
- `TrackId` 是辅助证据，不是最终决策权威；丢失轨迹不能输出控制目标。

## 配置替换

移除旧的 `target_selection_class_ratio`，替换为可归一化的六项权重：

- `target_selection_distance_weight`
- `target_selection_class_weight`
- `target_selection_confidence_weight`
- `target_selection_size_weight`
- `target_selection_continuity_weight`
- `target_selection_motion_weight`

关联新增：

- `tracker_class_cost_weight`：相同 cls 代价为 0，跨 cls 代价为 1；与位置、IoU、尺度一起归一化。
- `target_selection_motion_horizon_ms`：只用于评估目标在短时未来是否更接近准星，不改变控制输出瞄点。

旧 schema 16 显式或缺省的旧比例都会迁移为等价的类别/距离两项权重，其余新信号置零；损坏值继续失败关闭。新建配置使用六项推荐默认值。所有新字段继续支持运行期热更新。

前端 `targetClassPolicy` 模块按配置文件保存内部优先级、过滤与瞄点角色；切换配置时会把所选策略压平到后端实际消费的运行字段，`raw cls` 本身不被改写。

## 行为测试顺序

1. RED→GREEN：强几何连续的 `cls` 变化保留 Track ID；合理同类匹配仍优先于跨类匹配。
2. RED→GREEN：旧目标丢失但已有确认候选时，在切换延迟后接管，不等待完整 lost grace。
3. RED→GREEN：confidence、尺度、连续性和运动趋势能改变目标排序。
4. 配置迁移、校验、runtime compose、API schema 与 Web 参数模型贯通。
5. 前端显示六项权重、跨类关联说明和三层模块职责。
6. 运行核心、配置/API、Web 单测与构建；最后在 Jetson 做 Rust 测试和生产构建，不连接物理输出。

## 验证结果

- Rust 全工作区测试通过。
- Rust 全工作区 Clippy `-D warnings` 通过。
- Web 23 个测试文件、63 个测试通过，TypeScript 检查与生产构建通过。
- 固定起点审查发现的配置迁移、无效运动状态和类别配置切换问题均已修复并复核通过。

## 明确不做

- 不把游戏或模型专属类别名称硬编码进 Rust。
- 不新增 ReID 模型或外观特征网络。
- 不让低置信度框单独驱动控制；本次只替换现有检测批次后的决策逻辑。
- 不改变 DeepStream、TensorRT、NMS、kmNet 或控制输出协议。
