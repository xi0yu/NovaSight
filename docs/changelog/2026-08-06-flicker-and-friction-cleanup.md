# 2026-08-06 闪烁与冗余交互清理

针对 Studio 控制台在 WebSocket 高频 partial 帧期间产生的视觉闪烁、以及用户感知到的冗余确认 / 全局禁用按钮等问题，本轮系统性清理。共 20 项。

所有改动通过 `tsc --noEmit` 与 `cargo test -p novasight-runtime --test runtime_config` 验证。

---

## P0 — 闪烁根因

### P0-A · setConfigDraft 同值短路 + 拖动期间不同步
- 位置：`web/src/features/studio/StudioConsoleView.tsx:1333-1341`
- 问题：`useEffect([runtimeConfig])` 每帧 `setConfigDraft(next)`，触发整组件重渲染
- 修复：同值短路用 `runtimeConfigValuesEqual`；引入 `draggingControlId` state，编辑期间不同步
- 可见效果：所有受控参数控件在用户编辑期间不再被 partial 帧覆盖

### P0-B · SliderNumberControl 编辑期间 draftValue 优先
- 位置：`web/src/features/studio/StudioControls.tsx:60-211`
- 问题：拖动期间父层 prop 变化导致滑块视觉回弹
- 修复：`sliderValue` 编辑期用 `draftValue` 优先；`sliderMin/Max` 在编辑开始时快照冻结；commit 后用最新外部值再校准；向上暴露 `onEditingChange` / `controlId`
- 可见效果：拖动滑块不再回弹到服务端值

### P0-C · productConfigProfile memo deps 拆原子字段
- 位置：`web/src/features/studio/StudioConsoleView.tsx:1725-1800`
- 问题：`useMemo` 把 `artifact`（对象整体）放进 deps，每次后端重建 `ActiveModelDeployment` 都失效
- 修复：拆为 `activeArtifactId / VersionId / Kind / Path / Status` 五个原子字段
- 可见效果：`ProductConfigProfilePanel` 顶部 tone class 不再翻面

---

## P1.x — 性能与稳定性

### P1 · aimRoleRatios useMemo
- 位置：`web/src/features/studio/StudioConsoleView.tsx:1251-1264`
- 问题：每次 `runtimeConfig` 变化都生成新对象，下游 `===` 比对失效
- 修复：包 `useMemo` 稳定引用
- 可见效果：aim 拖动期间不会被父层 prop 覆写

### P1.6 · applyRuntimeFrame payload 短路
- 位置：`web/src/App.tsx:225-280`
- 问题：每个 partial 帧（即使无变化）都 setState 触发整树重渲染
- 修复：浅比较 incoming payload 与 current；相同则跳过 setState 和 `lastUpdated` 推进
- 可见效果：空闲状态下 paint 次数显著降低

### P1.7 · AimTargetRange onPointerCancel 不强回弹
- 位置：`web/src/features/studio/AimTargetRange.tsx:137-141`
- 问题：拖动中断时强制 `setDraft(ratios)`，用户当前位置被服务端值覆盖
- 修复：只释放 pointer capture，不再重置 draft
- 可见效果：拖动中断保留当前位置

---

## Menu A — 可达性与一致性

### A.1 · confirmationBusyRef 同步互斥
- 位置：`web/src/features/studio/StudioConsoleView.tsx:968-988`
- 问题：双击"开启输出"等危险确认会发两次请求（state-based busy 在同 tick 还未更新）
- 修复：`confirmationBusyRef` 同步互斥
- 可见效果：危险动作的二次点击只触发一次

### A.2 · 三弹窗 RAF 取消
- 位置：`web/src/features/studio/StudioConsoleView.tsx:1041-1130`
- 问题：弹窗打开后立即 Escape，下一帧 focus 抢回已关闭弹窗
- 修复：保存 `focusFrame` id，cleanup `cancelAnimationFrame`
- 可见效果：快速关闭弹窗后焦点不再被抢回

### A.3 · Stepper/Text/Inline isEditing 对称
- 位置：`web/src/features/studio/StudioControls.tsx`（Stepper / TextControl / InlineNumberControl / InlineTextControl）
- 问题：Slider 有 isEditing 守卫，兄弟控件没有 → 编辑期间被 partial 帧冲掉
- 修复：补齐 `onEditingChange` 协议 + `lastExternalValueRef` 跟踪 prop 变化
- 可见效果：所有文本类控件编辑期间稳定

### A.4 · 阶段/动作/建议三段式错误
- 位置：`web/src/features/studio/useMainlineLaunch.ts:150-160, 408, 530, 605`
- 问题：用户看到"启动失败"但不知道在哪一步、做什么、怎么试
- 修复：`formatLaunchError({stage, action, detail, hint})` 助手
- 可见效果：错误信息包含 stage / action / hint 三段

---

## Menu B — 数据正确性

### B.1 · runtimeConfig 写入序列保护
- 位置：`web/src/features/studio/StudioConsoleView.tsx:765-790` + 8 个写入路径
- 问题：慢响应可能覆盖新写入请求
- 修复：`beginRuntimeConfigWrite` / `finalizeRuntimeConfigWrite(seq, config)` 助手
- 覆盖：`saveConfigDialog` / `updateConfigField` / `updateConfigSection`（含 revision 冲突恢复）/ `setKmNetConnection` 输出门关 / 配置导入（3 个分支）
- 可见效果：用户改 A 同时改 B，旧响应不会污染 B 的结果

### B.2 · 保存期间阻止同一弹窗重开
- 位置：`web/src/features/studio/StudioConsoleView.tsx:916`
- 问题：保存响应回来时如果用户已重新打开同弹窗，旧响应会污染新弹窗
- 修复：`saveConfigDialog` 进入时 `pendingConfigWritesRef.current += 1`
- 可见效果：保存期间 `openConfigDialog` 内部守卫触发，不开新弹窗

### B.4 · useMainlineLaunch AbortController
- 位置：`web/src/features/studio/useMainlineLaunch.ts:208, 432, 599`
- 问题：取消 / 卸载后 `getRuntimeState` 仍可能继续到 15s 超时
- 修复：每个启动流程配 AbortController，cancel / cleanup 时 abort；post-cancel 确认 fetch 用独立 controller
- 可见效果：取消立即停止轮询

---

## B.3 lite — 事务状态机

### B.3 lite · applying 状态消除事务窗口闪烁
- 位置：`web/src/features/studio/productConfigProfile.ts` + `StudioConsoleView.tsx:1418-1430` + 4 个调用点
- 问题：写入期间 `configRestartRequired` 短暂 true → 面板翻"等待重启"再回"已生效"
- 修复：新增 `applying` 状态；`pendingApplyTick` state 镜像 `pendingConfigWritesRef`；`roiApplyLabel` / `postprocessApplyLabel` / `reload` item 优先显示"正在应用…"
- 可见效果：Profile 面板 / ROI / 后处理指示器在事务期间显示"正在应用…"，不再翻面

---

## C.3 — 顶栏时间戳节流

### C.3 · lastUpdated 节流 + memo 时间节点
- 位置：`web/src/App.tsx:185-220`
- 问题：每个 partial 帧都 `formatTime` 并 repaint 时间戳
- 修复：`LastUpdatedText` memo + 1Hz `setTimeout` 节流
- 可见效果：时间戳每秒最多 paint 1 次

---

## 确认对话框优化

### Conf 1 · 移除"放弃修改"确认
- 位置：`web/src/features/studio/StudioConsoleView.tsx:950-979`
- 问题：用户已点"关闭"，再问"你确定要关闭？"是冗余
- 修复：取消直接恢复 baseline + 关闭，弹 info toast "已放弃修改"
- 可见效果：取消 = 取消

### Conf 3 · 移除"清除准星模板"确认
- 位置：`web/src/features/studio/StudioConsoleView.tsx:2789-2803`
- 问题：模板可重学（F8），确认是过度保护
- 修复：直接调 `performClearCrosshair`，弹 info toast "按 F8 重新采样"
- 可见效果：清除 = 清除 + 提示恢复路径

### Conf 4 · kmNet 推荐值字段级勾选
- 位置：`web/src/features/studio/ActionConfirmationDialog.tsx`（新增 `fields` 可选 prop）+ `StudioConsoleView.tsx:3150-3195`
- 问题：原"一刀切覆盖全部字段"强迫用户接受 all-or-nothing
- 修复：弹窗内 5 行（host/port/uuid/monitor_port/auto_connect）每行带"旧值 → 新值"显示 + checkbox；默认只勾选与推荐值不同的字段
- 可见效果：可保留部分自己的参数

---

## D.x — 全局禁用按钮细化

### D.1 · 打开配置弹窗按钮细化
- 位置：`web/src/features/studio/StudioConsoleView.tsx:4152, 4202, 4387, 4397, 4412`
- 问题：`disabled={busy !== null}` 在 kmNet 连接 / 主链启动期间禁用所有弹窗打开按钮
- 修复：`disabled={configDialogSaving}`（与 `openConfigDialog` 内部守卫对齐）
- 可见效果：用户在 kmNet 连接期间仍可打开任意配置弹窗预览

### D.2 · Aim 拖动按钮细化 + memo
- 位置：`web/src/features/studio/StudioConsoleView.tsx:5124` + `AimTargetRange.tsx`
- 问题：Aim 拖动在 kmNet 连接期间被禁用；AimTargetRange 每次 partial 帧都重渲染
- 修复：同上细化；AimTargetRange 包 `React.memo`
- 可见效果：拖动不再被无关操作阻塞；aim 面板不重渲染

### D.3 · 类配置弹窗内文本输入细化
- 位置：`web/src/features/studio/StudioConsoleView.tsx:5058, 5091`
- 问题："新建配置名" / "当前配置名"输入在 kmNet 连接期间被禁用
- 修复：`disabled={configDialogSaving}`
- 可见效果：输入无副作用的操作不被全局阻塞

### D.4 · Body 滚动锁集中管理
- 位置：`web/src/features/studio/dialogFocus.ts`（新增 `acquireBodyScrollLock` / `releaseBodyScrollLock`）
- 问题：6+ 对话框各自保存/恢复 `body.style.overflow`，叠加时 cleanup 顺序错乱
- 修复：模块级计数器，first-open 保存原值，last-close 恢复
- 可见效果：错误中心在配置弹窗上叠加时，关闭顺序不再导致 scroll 锁卡住

### D.6 · 顶栏"写入中"指示器
- 位置：`web/src/features/studio/StudioConsoleView.tsx:3568-3576`
- 问题：用户不知道配置正在写（"正在应用…"只藏在 Profile 面板内部）
- 修复：顶栏条件渲染 `pendingApplyTick > 0` 时显示"写入中…"
- 可见效果：用户在顶栏就能看到配置写入状态

### D.7 · Aim 键盘 stale closure 修复
- 位置：`web/src/features/studio/AimTargetRange.tsx`
- 问题：按住 Arrow 时 `handleKey` 读 `draft[role]` 闭包，30Hz 触发同一闭包，滑块视觉跳回
- 修复：`draftRef` 镜像最新 draft；`handleKey` 从 ref 读
- 可见效果：键盘连续调节平滑无跳变

---

## 保留的二次确认（按原则保留）

按四条原则任一成立：
1. 不可逆 / 破坏性
2. 隐藏的物理副作用（→ kmNet 硬件）
3. 长时间服务中断
4. 大范围覆盖

- **开启物理鼠标输出**（→ kmNet 真实硬件）— 隐藏副作用
- **整份 JSON 配置导入** — 大范围覆盖
- **运行中切换模型**（stop pipeline / deploy / restart）— 长时间服务中断

---

## 未做的项目（建议优先级）

| 项 | 影响 | 建议 |
|---|---|---|
| C.1 `StudioConsoleView` 拆分 | 高 | 先用 React Profiler 找真正热点再拆 |
| C.2 productConfigProfile 静态/动态拆 | 中 | 边际收益已小（P0-C + P1.6 已显著稳定） |
| D 模型目录虚拟化 | 中 | 模型数量 < 100 时不需要 |
| Toast / Dialog 时序协同 | 中 | 需观察实际用户反馈 |
