# 算法参数体验与测试审查 · 2026-10-09

## 已实现

| 参数 | 接受范围 | 保存与运行时 |
| --- | --- | --- |
| 搜索半径 | 0～1000 px | 前端、API schema、配置校验与目标筛选统一 |
| 触发等待 | 0～2000 ms | 保存时同步等待开关，运行时使用该时长 |
| 力度渐增 | 0～1000 ms | 建议 200～600 ms，0 表示立即完整力度 |
| X/Y 单次输出上限 | 每轴 0～1000 counts | 发送前、开火稳定补偿后分别限幅；0 禁止该轴移动 |

- 删除搜索范围的解释列表、判定详情、模拟目标与预览，以及每个自动保存控件旁的重复提示；自动保存和失败回执保留。
- 启动响应使用后端同一 smoothstep 定义 `q²(3−2q)`，先等待，再从零升至完整力度。图形跟随输入草稿立即更新，曲线标出等待和渐增时长。等待结束时已有有效目标是示意前提；目标晚到时，真实渐增起点也会晚到。
- 进阶响应试算也包含触发等待，等待期间输出为 0。
- 移动手感增加“投影误差 → Kp 响应 → X/Y 限幅”的图标与导线结构，展示当前配置下的单次 Atan 浮点需求和正负轴向限幅。图是按帧、完整力度示意，未计预测、整数余量和物理响应，不向设备发送命令。
- 算法页常驻当前模型与“管理模型”入口；设备页已有同名入口，首页的缺模型任务入口继续保留。模型页统一暖色卡片，删除重复的部署流程条；文件搜索、筛选、排序、验证、部署、草稿保护、文件整理与真实装载状态仍保留。
- 浏览器回归找到了“X 轴提交后，Tab 聚焦 Y 轴就阻止保存”的问题。保存调度现在只处理已提交的页面草稿，不因下一控件获得焦点而延后；未提交的局部输入依旧保留。
- 移除与上述删掉元素对应的 83 条废弃样式或组合选择器分支。
- 参数预览明确提示“预览修改已保留、不写入设备文件”，避免显示永远等待的自动保存或声称已与设备同步。

## 六项测试审查结论

1. **参数化**：模型按文件名、项目、标签搜索合成同一行为表；运行错误翻译合成消息表。采集差异、故障来源、Toast 配色/时限等已有的参数化行为保留。参数化减少重复代码，展开后的案例仍计入运行数量。
2. **删除重复或无独立行为的检查**：去掉静态挂载来重复证明控件不存在的测试、旧准星学习元素的残留检查，以及只核对内部 CSS 变量的几何测试。筛选的摘要与禁用行为并入真正的搜索/筛选流程，排序记忆并入会话恢复流程。浏览器重复的通用页面宽度/导航测试并入既有真实任务和 Utility Dialog 流程；固定 375px 的回归只运行一次。
3. **改为行为验证**：删除直接读取内部确认对象的 Hook 用例，由真实登记确认 Dialog 的组件集成及浏览器交互覆盖。模型元数据部分成功、部署成功后刷新失败改为点击操作并核对可见文件、错误或实际 ModelSwitchDialog。鼠标协议不再绑定 DOM 子节点类名和人为构造的调用顺序。未发现反射调用私有方法的测试；对公共 API 请求、输出同意标记、重复请求数的断言保留，因为它们验证服务边界和用户授权。
4. **分层**：整页配置/草稿/原子保存、错误中心、模型工作流、API 请求和真实 Vite 代理归为 `*.integration.test.*`，使用独立 Vitest 配置。纯解析、投影、门控、数学与单个组件行为保留单测。实际焦点、滚动、尺寸、对比度、文件选择和 CSS 进度条留在浏览器层。CI 和 Jetson 验收脚本已改跑 `test:ci`，包含两层及浏览器测试类型检查。
5. **补边界与异常**：五个参数 0、最大值、负数、超过最大值、NaN/Infinity；搜索圆半径 0 的圆心准入/偏离拒绝；渐增超过 1000 的拒绝；触发等待与渐增 0/非零组合；有符号限幅、单轴关闭、双轴关闭和运行中调整；旧 schema 不能扩大前端范围；非法数字草稿不能发起保存；保存某字段时不能覆盖下一字段的未完成文本。原有失败保留草稿、并发保存 revision 重基、授权到期、紧停因果、物理输出确认和敏感数据遮蔽未删。
6. **完整文件交付**：浏览器用例按鼠标协议、算法与模型、运行/授权/日志、桌面体验拆成四个业务文件，共享 API fixture；授权首次进入仍有独立文件。所有测试完整文件、fixture、配置和执行命令会打包到 `out/frontend-tests-refactored.zip`。

## 验证

- 原前端 210 项：整理后 **154 单测 + 51 集成 = 205 项**，无跳过；当前完整运行见 `out/algorithm-test-refactor-ci.log`。
- 浏览器原 66 项，整理后 **62 项全部通过**（桌面 Chromium 与宽度敏感的 mobile-safety）；无跳过、无重试转绿。结果见 `out/algorithm-test-refactor-browser.json`。
- Rust `novasight-config/core/pipeline/api --no-default-features` 共 191 项通过。输出限幅通过 RecordingPointerDevice 的最终发送回执验证，未接物理设备。
- 产品与浏览器测试类型检查、生产构建、图标注册/11 组 token 对比度检查通过。

## 逐文件审查清单

| 完整测试文件 | 层 | 展开案例 | 处理 |
| --- | --- | ---: | --- |
| `web/src/lib/toast.test.tsx` | 单测 | 15 | 保留独立业务、协议或安全行为 |
| `web/src/contracts/activity.test.ts` | 单测 | 3 | 保留独立业务、协议或安全行为 |
| `web/src/contracts/capture.test.ts` | 单测 | 2 | 保留独立业务、协议或安全行为 |
| `web/src/contracts/license.regression-1.test.ts` | 单测 | 1 | 保留独立业务、协议或安全行为 |
| `web/src/contracts/modelPublish.test.ts` | 单测 | 2 | 保留独立业务、协议或安全行为 |
| `web/src/contracts/runtimeStatusMessage.test.ts` | 单测 | 2 | 保留独立业务、协议或安全行为 |
| `web/src/contracts/studioLayout.test.ts` | 单测 | 4 | 保留独立业务、协议或安全行为 |
| `web/src/features/activity/ActivityView.test.tsx` | 单测 | 5 | 保留独立业务、协议或安全行为 |
| `web/src/features/activity/useActivityStream.test.tsx` | 单测 | 1 | 本轮整理、行为改写或补边界 |
| `web/src/features/auth/AuthGate.test.tsx` | 单测 | 5 | 保留独立业务、协议或安全行为 |
| `web/src/features/license/LicensePanel.test.tsx` | 单测 | 7 | 保留独立业务、协议或安全行为 |
| `web/src/features/models/ModelCatalogTree.test.tsx` | 单测 | 2 | 保留独立业务、协议或安全行为 |
| `web/src/features/models/ModelSelectionPanel.test.tsx` | 单测 | 21 | 本轮整理、行为改写或补边界 |
| `web/src/features/shared/format.test.ts` | 单测 | 4 | 本轮整理、行为改写或补边界 |
| `web/src/features/runtime/SafetyOperationContext.test.tsx` | 单测 | 4 | 保留独立业务、协议或安全行为 |
| `web/src/features/runtime/runtimeProjection.test.ts` | 单测 | 11 | 保留独立业务、协议或安全行为 |
| `web/src/features/runtime/runtimeSnapshotGate.test.ts` | 单测 | 5 | 保留独立业务、协议或安全行为 |
| `web/src/features/studio/AlgorithmResponsePreview.test.tsx` | 单测 | 2 | 本轮整理、行为改写或补边界 |
| `web/src/features/studio/MouseProtocolView.test.tsx` | 单测 | 14 | 本轮整理、行为改写或补边界 |
| `web/src/features/studio/PredictionInsight.test.tsx` | 单测 | 5 | 本轮整理、行为改写或补边界 |
| `web/src/features/studio/ProductJourneyViews.test.tsx` | 单测 | 9 | 保留独立业务、协议或安全行为 |
| `web/src/features/studio/RuntimeOverviewView.test.tsx` | 单测 | 10 | 保留独立业务、协议或安全行为 |
| `web/src/features/studio/StudioControls.test.tsx` | 单测 | 4 | 本轮整理、行为改写或补边界 |
| `web/src/features/studio/StudioNavigation.test.tsx` | 单测 | 4 | 保留独立业务、协议或安全行为 |
| `web/src/features/studio/StudioRuntimeBar.test.tsx` | 单测 | 2 | 本轮整理、行为改写或补边界 |
| `web/src/features/studio/TargetClassEditor.test.tsx` | 单测 | 1 | 保留独立业务、协议或安全行为 |
| `web/src/features/studio/TargetRangeControls.test.tsx` | 单测 | 1 | 本轮整理、行为改写或补边界 |
| `web/src/features/studio/algorithmParameterModel.test.ts` | 单测 | 2 | 本轮整理、行为改写或补边界 |
| `web/src/features/studio/configDiff.test.ts` | 单测 | 1 | 保留独立业务、协议或安全行为 |
| `web/src/features/studio/controlTrace.test.ts` | 单测 | 1 | 保留独立业务、协议或安全行为 |
| `web/src/features/studio/useMainlineLaunch.test.ts` | 单测 | 2 | 保留独立业务、协议或安全行为 |
| `web/src/features/studio/useModelSwitchWorkflow.test.ts` | 单测 | 2 | 本轮整理、行为改写或补边界 |
| `web/src/viteProxy.integration.test.mjs` | 集成 | 2 | 本轮整理、行为改写或补边界 |
| `web/src/contracts/modelRequests.integration.test.ts` | 集成 | 1 | 本轮整理、行为改写或补边界 |
| `web/src/features/studio/StudioConfig.integration.test.tsx` | 集成 | 42 | 本轮整理、行为改写或补边界 |
| `web/src/features/studio/StudioErrors.integration.test.tsx` | 集成 | 4 | 本轮整理、行为改写或补边界 |
| `web/src/features/studio/useModelSwitchWorkflow.metadata.integration.test.tsx` | 集成 | 2 | 本轮整理、行为改写或补边界 |

## 桌面预览与截图

- `http://127.0.0.1:7351/preview.html?page=params`：范围/触发、移动输出、目标类别与常驻模型入口。
- `http://127.0.0.1:7351/preview.html?page=models`：模型工作区。
- `out/algorithm-startup-after.png`：等待 200 ms + 渐增 400 ms 的实际界面。
- `out/algorithm-output-after.png`：Kp 0.2、X 上限 8、Y 上限 0 的实际界面。
- `out/algorithm-models-after.png`：模型管理入口与空目录界面。

## 执行命令

在当前仓库：`pnpm --dir web test:ci`、`pnpm --dir web test:e2e`、`pnpm --dir web build`、`pnpm --dir web visual:audit`。

Rust 验证：`cargo test -p novasight-config -p novasight-core -p novasight-pipeline -p novasight-api --no-default-features`。

测试包适用于当前 NovaSight 仓库，包含完整测试文件、fixture 和配置，不是独立应用源码包。

## 范围与限制

本文初次整理时 HEAD 为 99f98b34，尚未提交、部署或修改现有用户配置文件。新范围是严格校验：已有文件若使用大于新上限的值，读取/导入/启动时可能被拒绝，需要调整到新范围，不静默改写文件。

搜索半径为 0 只允许与准星重合的当前瞄点，并不被解释为“禁用整个算法”。输出轴上限为 0 才禁止该轴最终移动。曲线是配置示意，不能作为实机移动轨迹、收敛性能或硬件校准的证明。未连接 kmBox/Jetson 验证物理输出；高频实机负载未测。

## 2026-10-10 提交前补充

- 单次输出图用琥珀色加粗区段标记真实限幅位置。纵轴按 X/Y 输出上限固定，调大 Kp 后限幅平台不向横轴移动；灰色未限幅需求超出图表范围时裁切。
- 开火稳定页与标题统一为 1100 px 居中，精简重复文案，工作原理默认折叠。预览切换不会显示无法完成的自动保存等待。
- `preview.html` 不写配置文件，刷新恢复模拟默认值属于预期行为。正式参数保存增加 Kp 与开火稳定强度重新读取、重新挂载的行为回归；后端持久化测试验证 YAML 回读。
- 新参数写入接口补齐 `config_write` 授权，受限许可证返回 403 且配置文件、revision 不变。临时移除授权检查时回归失败，恢复后验证。
- 当前前端验证为 156 项单测、52 项集成测试、62 项浏览器回归；类型检查、生产构建及图标/对比度审查通过。前文 154/51 是初次重构时的记录。
- 补充截图：`out/algorithm-output-fixed-axis.png`、`out/fire-stability-centered.png`。截图与验证日志属于本地证据，不纳入源码提交。
- 提交前全工作区 Rust 验证：323 项测试通过，格式和严格 Clippy 通过。迁移回归同步接受搜索半径 0/1000，并验证超界写入拒绝。
