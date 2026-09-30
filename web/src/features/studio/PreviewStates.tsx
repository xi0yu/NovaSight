import { useRef, useState } from "react";
import type { ModelCatalogModel, LicenseStatus } from "../../api";
import { ToastHost } from "../../components/ToastHost";
import { pushToastRaw } from "../../lib/toast";
import { ActivityView, type ActivityItem } from "../activity/ActivityView";
import { AuthGate } from "../auth/AuthGate";
import { LicensePanel } from "../license/LicensePanel";
import { ModelManagerDialog } from "../models/ModelManagerDialog";
import { ModelSwitchDialog } from "../models/ModelSwitchDialog";
import { SafetyOperationProvider, useSafetyOperation } from "../runtime/SafetyOperationContext";
import { ActionConfirmationDialog } from "./ActionConfirmationDialog";
import { SettingsView } from "./SettingsView";
import { FrontendPreview, previewRuntime } from "./FrontendPreview";
import "./preview-states.css";

const examples: Array<[string, string, string]> = [
  ["pages", "完整页面", "默认页面与真实组件；预览不连接设备，保存操作会被拒绝。"],
  ["offline", "首页 · 服务断连", "没有运行快照时，不展示零值、不声称输出安全。"],
  ["stale", "首页 · 状态过期", "保留最近读数，但不把旧的开关状态当作实时状态。"],
  ["activity", "日志 · 搜索与清理失败", "可以搜索、筛选和展开详情；点击清理可查看失败后保留记录的状态。"],
  ["settings", "设置 · 版本待生效", "保存版本与运行版本不一致时，显示等待生效。"],
  ["settings-unavailable", "设置 · 配置未读取", "无法确认版本时，禁用备份与恢复，不显示版本一致。"],
  ["models", "模型库 · 文件与标签", "可以筛选文件、编辑标签和展开文件操作；关闭时保护未保存的标签。"],
  ["models-empty", "模型库 · 空目录", "没有模型时提示支持的文件类型和放置位置。"],
  ["models-stale", "模型库 · 刷新失败", "保留上次读取的文件并标记过期，暂停模型切换。"],
  ["switch-running", "模型切换 · 等待回执", "只展示已取得的回执，不编造设备内部进度百分比。"],
  ["switch-success", "模型切换 · 成功", "切换结果已确认；后续刷新失败不能抹掉这个结果。"],
  ["switch-failed", "模型切换 · 结果未确认", "请求失败不等于操作没有执行，保留核对建议。"],
  ["confirmation", "确认 · 未保存修改", "关闭编辑窗口或恢复备份前出现。取消返回编辑，不丢草稿。"],
  ["confirmation-error", "确认 · 操作失败", "失败原因留在操作附近，允许重试或返回，不重复弹出多个错误。"],
  ["auth", "登录 · 授权与失败", "仅输入演示文本；提交不会发送到服务器，可查看表单失败状态。"],
  ["license", "授权 · 状态与更换", "展开更换授权、排查信息，或查看退出授权的确认提示。"],
  ["safety", "安全 · 停止回执", "操作请求和设备已确认是不同状态；本页按钮仅模拟回执。"],
  ["toast", "通知 · 成功与失败", "短消息用于结果反馈，排查编号按需展开。"],
];
const sampleModel: ModelCatalogModel = { type: "model", name: "arena-fp16.engine", relative_path: "arena-fp16.engine", kind: "engine", size_bytes: 18400000, scan_status: "need_confirm", scan_reason: "", recommendation: "recommended", tags: ["稳定", "低延迟"] };
const sampleActivity: ActivityItem[] = [
  { key: "1", title: "画面规格已保存", detail: "1920 × 1080 · 120 FPS，下次启动使用新规格。", tag: "采集", tone: "success", time: Date.now() - 15000 },
  { key: "2", title: "连接曾中断", detail: "此前未能读取设备状态；此记录不代表当前仍然断连。", tag: "连接", tone: "warn", time: Date.now() - 120000 },
  { key: "3", title: "模型读取失败", detail: "请检查模型文件是否仍在原来的目录。", tag: "模型", tone: "error", time: Date.now() - 300000, technicalDetail: "ENOENT: models/arena-fp16.engine", requestId: "preview-request-001" },
];
const demoLicense = { configured: true, valid: true, tier: "pro", features: ["capture", "runtime", "models"], credential_format: "jwt_rs256", license_id: "preview-license", temporary_access_supported: true, expires_at: null } as LicenseStatus;

function SafetyPreview() {
  const safety = useSafetyOperation();
  return <div className="preview-inline-actions">
    <button className="console-button" onClick={() => safety.beginEmergencyStop(null)}>模拟等待回执</button>
    <button className="console-button" onClick={() => safety.markEmergencyStopUnconfirmed("未收到设备确认。请检查设备连接，不能仅凭按钮已点击就认为输出已停止。")}>模拟未确认</button>
    <button className="console-button" onClick={() => { safety.beginEmergencyStop(previewRuntime); safety.reconcileEmergencyStop({ ...previewRuntime, semantic: { ...previewRuntime.semantic, snapshot_sequence: 2 } }); }}>模拟已确认</button>
  </div>;
}

function StateExample({ name }: { name: string }) {
  const [open, setOpen] = useState(true);
  const [error, setError] = useState(name === "confirmation-error" ? "设备暂时无法保存。修改仍保留在当前页面。" : "");
  const [model, setModel] = useState(sampleModel);
  const [dirty, setDirty] = useState(false);
  const [discard, setDiscard] = useState(false);
  const [notice, setNotice] = useState("");
  const dialogRef = useRef<HTMLElement>(null);
  const close = () => setOpen(false);
  if (name === "pages" || name === "offline" || name === "stale") return <FrontendPreview scenario={name} />;
  let content;
  if (name === "activity") content = <ActivityView items={sampleActivity} onClear={() => { throw new Error("当前身份无权清理设备历史记录。"); }} onOpenDetails={() => setNotice("展开每条记录的“原始错误与开发者详情”查看证据。")} />;
  else if (name.startsWith("settings")) content = <SettingsView modules={["backup", "restore", "details"]} desiredRevision={12} effectiveRevision={11} restartRequired={false} configAvailable={name === "settings"} runtimeVerified={name === "settings"} operationPending={false} parameterChangesPending={false} onExport={() => setNotice("此处只预览备份入口，不生成设备配置。")} onImport={() => setNotice("文件未上传；正式页面会先显示变更确认。")} onNavigate={() => {}} />;
  else if (name.startsWith("models")) content = <>
    <ModelManagerDialog activeModelName="未部署模型" open={open} onClose={() => dirty ? setDiscard(true) : close()} panelProps={{
      root: { type: "directory", name: "models", relative_path: "", children: name === "models-empty" ? [] : [model] }, loading: false, directoryCount: 1, modelCount: name === "models-empty" ? 0 : 1,
      selectedPath: model.relative_path, selectedModel: name === "models-empty" ? null : model, selectedArtifact: null, selectedVersion: null,
      activeArtifactId: null, activeArtifactPath: "", runtimeBackend: "", runtimeInputShape: "", catalogMessage: "", catalogError: name === "models-stale" ? "设备连接中断，未取得最新目录。" : null, switchMessage: "", busy: null, canSwitch: true, parserPreset: "auto",
      onParserPresetChange: () => {}, onRefresh: () => setNotice("预览目录未变化。"), onCreateFolder: async () => { throw new Error("预览不会创建设备文件夹。"); }, onRequestMove: () => setNotice("预览不会移动设备文件。"), onSelectModel: setModel,
      onDraftChange: setDirty, onSaveMetadata: (recommendation, tags) => { setModel({ ...model, recommendation, tags }); setNotice("标签已保存在本次预览中，刷新后重置。"); }, onSwitch: () => setNotice("请从状态选择器查看模型切换流程；未执行真实切换。"),
    }} />
    {discard ? <ActionConfirmationDialog request={{ eyebrow: "模型整理", title: "放弃未保存的模型标签？", description: "取消返回编辑；放弃只会清除本次预览草稿。", confirmLabel: "放弃修改", danger: true, onConfirm: close }} busy={false} onCancel={() => setDiscard(false)} onConfirm={() => { setDiscard(false); close(); }} /> : null}
  </>;
  else if (name.startsWith("switch-")) content = <ModelSwitchDialog open={open} dialogRef={dialogRef} modelName={sampleModel.name} status={name === "switch-running" ? "running" : name === "switch-success" ? "success" : "failed"} completedStages={name === "switch-success" ? 3 : 2} currentStage={2} detail={name === "switch-success" ? "设备已确认切换。最新状态读取失败，请重新连接后核对。" : "等待设备完成验证与切换；尚无最终回执。"} error={name === "switch-failed" ? "请求超时。请核对当前模型和运行状态，不能假定切换未生效。" : ""} onClose={close} />;
  else if (name.startsWith("confirmation")) content = open ? <ActionConfirmationDialog request={{ eyebrow: "配置修改", title: "放弃未保存的修改？", description: "当前编辑尚未保存到设备。取消可继续编辑，设备仍使用原来的配置。", details: ["搜索半径：180 → 240 px", "触发延迟：0 → 100 ms"], confirmLabel: "放弃修改", danger: true, onConfirm: close }} busy={false} error={error} onCancel={close} onConfirm={() => name === "confirmation-error" ? setError("设备暂时无法保存。修改仍保留在当前页面。") : close()} /> : null;
  else if (name === "auth") content = <AuthGate><p>授权完成</p></AuthGate>;
  else if (name === "license") content = <LicensePanel license={demoLicense} onLicenseChange={() => {}} />;
  else if (name === "safety") content = <SafetyOperationProvider><SafetyPreview /></SafetyOperationProvider>;
  else content = <div className="preview-inline-actions">
    <button className="console-button primary" onClick={() => pushToastRaw({ tone: "success", source: "preview", status: null, title: "设置已保存", detail: "设备已确认应用新的画面规格。" })}>展示成功通知</button>
    <button className="console-button" onClick={() => pushToastRaw({ tone: "error", source: "preview", status: null, title: "保存未确认", detail: "请检查连接。当前编辑仍然保留。", requestId: "preview-request-002" })}>展示失败通知</button>
  </div>;
  return <main className="preview-state-stage">
    <header><span>组件状态预览</span><h1>{examples.find(([id]) => id === name)?.[1]}</h1><p>{examples.find(([id]) => id === name)?.[2]}</p></header>
    {notice ? <p className="preview-local-notice" role="status">{notice}</p> : null}
    {!open ? <button className="console-button" onClick={() => setOpen(true)}>重新打开组件</button> : null}
    {content}<ToastHost />
  </main>;
}

export function PreviewStates() {
  const query = new URLSearchParams(location.search);
  const initial = query.get("showcase") ?? "pages";
  const [name, setName] = useState(examples.some(([id]) => id === initial) ? initial : "pages");
  return <div className="preview-root">
    <aside className="preview-tools" aria-label="界面预览工具">
      <strong><i />界面预览</strong><span>模拟数据 · 不连接设备</span>
      <label>状态<select aria-label="选择预览状态" value={name} onChange={(event) => {
        setName(event.target.value);
        const url = new URL(location.href); url.searchParams.set("showcase", event.target.value); history.replaceState(null, "", url);
        if (event.target.value === "offline" || event.target.value === "stale") { url.searchParams.set("page", "overview"); history.replaceState(null, "", url); }
      }}>{examples.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
    </aside>
    <StateExample key={name} name={name} />
  </div>;
}
