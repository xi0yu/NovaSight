import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { LoadedImage } from "../../components/ui";
import { reportError, reportInfo, reportSuccess } from "../../lib/toast";
import { NovaIcon } from "../../components/visual";
import { getErrorMessage } from "../shared/format";
import { useDialogInteraction } from "./dialogFocus";
import netImage from "../../assets/output-consumers/kmbox-net-official.png";
import netProImage from "../../assets/output-consumers/kmbox-net-pro-8k-official.png";
import nbImage from "../../assets/output-consumers/kmbox-nb-official.png";
import aiImage from "../../assets/output-consumers/kmbox-ai-official.png";
import emptyImage from "../../assets/output-consumers/empty-mouse.svg";
import "./mouse-protocol.css";

export type MouseProtocol = { id: string; model: "net" | "net_pro8k" | "nb" | "ai"; host: string; port: number; uuid: string };
const defaults = { host: "192.168.2.188", port: 8888, uuid: "A1B2C3D4" };
const models = {
  net_pro8k: { label: "kmBox NetPro8K", image: netProImage },
  net: { label: "kmBox Net", image: netImage },
  nb: { label: "kmBox NB", image: nbImage },
  ai: { label: "kmBox AI", image: aiImage },
};

export function readMouseProtocols(hardware: Record<string, unknown>): MouseProtocol[] {
  if (Array.isArray(hardware.mouse_protocols) && hardware.mouse_protocols.length > 0) return hardware.mouse_protocols.filter((item): item is MouseProtocol =>
    item !== null && typeof item === "object" && typeof item.id === "string" && typeof item.model === "string" && Object.prototype.hasOwnProperty.call(models, item.model)
    && typeof item.host === "string" && typeof item.port === "number" && typeof item.uuid === "string");
  // Preserve an existing connection until the user explicitly selects another protocol.
  return hardware.backend !== "null" && typeof hardware.host === "string" && typeof hardware.uuid === "string"
    && hardware.host && hardware.uuid ? [{ id: "legacy", model: "net", host: hardware.host,
      port: typeof hardware.port === "number" ? hardware.port : 8888, uuid: hardware.uuid }] : [];
}

export function MouseProtocolView({ protocols, selectedId, available, running, saving, previewOnly, diagnostics, onSave, onSelect, onProbe, onRemove, onGoHome }: {
  protocols: MouseProtocol[]; selectedId: string; available: boolean; running: boolean; saving: boolean; previewOnly: boolean;
  diagnostics?: ReactNode;
  onSave: (protocol: MouseProtocol) => Promise<void>;
  onSelect: (protocol: MouseProtocol | null) => Promise<void>;
  onProbe: (protocol: MouseProtocol) => Promise<{ elapsed_ms: number }>;
  onRemove?: (protocol: MouseProtocol) => Promise<void>;
  onGoHome?: () => void;
}) {
  const [editor, setEditor] = useState<{ id?: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const locked = !available || running || saving || busy;
  const unavailableReason = !available ? "配置尚未读取，请恢复服务连接。" : running ? "主链运行中，请先关闭首页运行开关。" : saving || busy ? "正在保存协议…" : "";
  const select = async (protocol: MouseProtocol | null) => {
    if (busyRef.current || locked || selectedId === (protocol?.id ?? "empty")) return;
    busyRef.current = true; setBusy(true);
    try { await onSelect(protocol); }
    catch { /* The shared configuration writer reports failures through the global toast. */ }
    finally { busyRef.current = false; setBusy(false); }
  };
  const editedProtocol = editor?.id ? protocols.find((protocol) => protocol.id === editor.id) : undefined;
  const configure = () => {
    const selected = protocols.find((protocol) => protocol.id === selectedId);
    if (!selected) {
      reportInfo(selectedId === "empty" ? "空协议无需配置" : "请选择要配置的协议", "点击其他协议卡片后，再配置连接参数。", "mouse-protocol");
      return;
    }
    setEditor({ id: selected.id });
  };
  return <section className="mouse-protocol-workspace" aria-label="鼠标协议管理" aria-busy={busy || saving}>
    <div className="mouse-protocol-toolbar">
      <button type="button" className="console-button mouse-protocol-configure" title={selectedId === "empty" ? "空协议无需配置" : "配置选中的协议"}
        disabled={!available || saving || busy} onClick={configure}>
        <NovaIcon name="settings" size={18} aria-hidden="true" />配置协议</button>
      <button type="button" className="console-button primary mouse-protocol-add" title={unavailableReason || (protocols.length >= 64 ? "最多保存 64 个协议，请打开未启用协议清理。" : "添加一个鼠标协议")}
        disabled={locked || protocols.length >= 64} onClick={() => setEditor({})}>
        <span aria-hidden="true">＋</span>新增协议</button>
    </div>
    <div className="mouse-protocol-cards" aria-label="可用鼠标协议">
      {[null, ...protocols].map((protocol) => {
        const id = protocol?.id ?? "empty";
        const model = protocol ? models[protocol.model] : null;
        const label = model ? `${model.label} · ${protocol!.host}:${protocol!.port}` : "空";
        const active = selectedId === id;
        return <article key={id} className="mouse-protocol-card" data-model={protocol?.model ?? "empty"} data-active={active} aria-label={label}>
          <button type="button" className="mouse-protocol-select" aria-label={label} aria-pressed={active}
            title={unavailableReason || (active ? "当前协议已启用" : protocol ? `选中并启用 ${label}` : "选中空协议，停止向物理鼠标输出")}
            disabled={locked} onClick={() => void select(protocol)}>
            <span className="mouse-protocol-picture"><LoadedImage src={model?.image ?? emptyImage} alt={model ? `${model.label} 官方实物图` : "空协议鼠标光标"} /></span>
            <span className="mouse-protocol-caption" title={label}>{model ? `${model.label.replace("kmBox ", "")} · ${protocol!.host}` : "空"}</span>
          </button>
          {active ? <span className="mouse-protocol-active-badge"><NovaIcon name="check-circle" size={16} aria-hidden="true" />已启用</span> : null}
        </article>;
      })}
    </div>
    {editor ? <ProtocolEditor protocols={protocols} onPick={(id) => setEditor({ id })} protocol={editedProtocol} disabled={locked} canProbe={!locked && !previewOnly} disabledReason={unavailableReason} running={running} previewOnly={previewOnly} selected={editor.id === selectedId} onGoHome={onGoHome} onRemove={onRemove}
      diagnostics={editedProtocol && editor.id === selectedId ? diagnostics : undefined}
      onProbe={onProbe} onClose={() => setEditor(null)} onSave={async (next) => {
        const duplicate = protocols.find((item) => item.id !== next.id && sameMouseProtocol(item, next));
        if (duplicate) {
          reportSuccess("协议已存在", "已打开相同协议，没有新增重复卡片。", "mouse-protocol", "feedback");
          setEditor({ id: duplicate.id });
          return false;
        }
        await onSave(next); return true;
      }} key={editor.id ?? "new"} /> : null}
  </section>;
}

export function sameMouseProtocol(left: MouseProtocol, right: MouseProtocol): boolean {
  return left.model === right.model && left.host === right.host && left.port === right.port && left.uuid.toUpperCase() === right.uuid.toUpperCase();
}

function ProtocolEditor({ protocols, onPick, protocol, disabled, disabledReason, running, previewOnly, selected, canProbe, diagnostics, onProbe, onClose, onSave, onRemove, onGoHome }: {
  protocols: MouseProtocol[]; onPick: (id: string) => void;
  protocol?: MouseProtocol; disabled: boolean; disabledReason: string; running: boolean; previewOnly: boolean; selected: boolean; canProbe: boolean; diagnostics?: ReactNode;
  onProbe: (protocol: MouseProtocol) => Promise<{ elapsed_ms: number }>;
  onClose: () => void; onSave: (protocol: MouseProtocol) => Promise<boolean>;
  onRemove?: (protocol: MouseProtocol) => Promise<void>; onGoHome?: () => void;
}) {
  const [draft, setDraft] = useState<Omit<MouseProtocol, "id">>(protocol ?? { model: "net_pro8k", host: "", port: NaN, uuid: "" });
  const draftChanged = protocol !== undefined && (draft.model !== protocol.model || draft.host !== protocol.host || draft.port !== protocol.port || draft.uuid !== protocol.uuid);
  const [id] = useState(() => protocol?.id ?? crypto.randomUUID());
  const [error, setError] = useState("");
  const [invalidField, setInvalidField] = useState<"host" | "port" | "uuid" | null>(null);
  const errorId = useId();
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  useEffect(() => {
    const changed = busy || draftChanged || (!protocol && (draft.host !== "" || !Number.isNaN(draft.port) || draft.uuid !== "" || draft.model !== "net_pro8k"));
    if (!changed) return;
    const protect = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", protect);
    return () => window.removeEventListener("beforeunload", protect);
  }, [busy, draft, draftChanged, protocol]);
  const [operation, setOperation] = useState<"save" | "probe" | "remove" | null>(null);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [result, setResult] = useState<number | null>(null);
  useEffect(() => { setResult(null); }, [draft, canProbe]);
  const dialogRef = useRef<HTMLElement>(null);
  useDialogInteraction(dialogRef, true, onClose, { blocked: busy, initialFocus: "select:not(:disabled)" });
  const updateDraft = (values: Partial<Omit<MouseProtocol, "id">>) => {
    setDraft((current) => ({ ...current, ...values }));
    setError(""); setInvalidField(null);
  };
  const fieldError = (field: "host" | "port" | "uuid") => invalidField === field && error
    ? <small className="protocol-field-error" id={errorId} role="alert">{error}</small> : null;
  const validatedDraft = () => {
    const reject = (field: "host" | "port" | "uuid", message: string) => {
      setError(message); setInvalidField(field);
      dialogRef.current?.querySelector<HTMLInputElement>(`[data-protocol-field="${field}"]`)?.focus();
      reportError(new Error(message), { source: "mouse-protocol-validation", title: "协议参数无效", tone: "error", popup: "feedback", publicDetail: message });
    };
    const next = { ...draft, host: draft.host.trim() || defaults.host, port: Number.isNaN(draft.port) ? defaults.port : draft.port, uuid: draft.uuid.trim() || defaults.uuid };
    const parts = next.host.split(".");
    if (parts.length !== 4 || parts.some((part) => !/^(0|[1-9]\d{0,2})$/.test(part) || Number(part) > 255)) { reject("host", "IP 需为有效的 IPv4 地址。"); return; }
    if (!Number.isInteger(next.port) || next.port < 1 || next.port > 65535) { reject("port", "Port 需为 1–65535 的整数。"); return; }
    if (!/^[\da-f]{8}$/i.test(next.uuid)) { reject("uuid", "UUID 需为 8 位十六进制字符。"); return; }
    return { ...next, id };
  };
  const perform = async (operation: "save" | "probe" | "remove", action: (next: MouseProtocol) => Promise<void>) => {
    if (busyRef.current || disabled) return;
    const next = operation === "remove" && protocol ? protocol : validatedDraft();
    if (!next) return;
    busyRef.current = true; setBusy(true); setOperation(operation); setError(""); setInvalidField(null); setResult(null);
    try { await action(next); }
    catch (error) { setError(getErrorMessage(error)); }
    finally { busyRef.current = false; setBusy(false); setOperation(null); }
  };
  const submit = () => perform("save", async (next) => { if (await onSave(next)) onClose(); });
  const testConnection = () => perform("probe", async (next) => {
    if (!canProbe) return;
    const response = await onProbe(next);
    setResult(response.elapsed_ms);
  });
  return <div className="action-confirmation-layer mouse-protocol-editor-layer" onMouseDown={(event) => { if (event.target === event.currentTarget && !busyRef.current) onClose(); }}><section className="mouse-protocol-editor" role="dialog" aria-modal="true" aria-labelledby="protocol-editor-title" ref={dialogRef} tabIndex={-1} aria-busy={busy}>
    <header><div className="mouse-protocol-editor-portrait" aria-hidden="true"><LoadedImage src={models[draft.model].image} alt="" /></div><div><h2 id="protocol-editor-title">{running ? "查看协议" : protocol ? "配置协议" : "新增协议"}</h2><p>{previewOnly ? "预览只更新本页。" : "保存后应用配置并写入设备文件。"}留空使用默认值。</p></div></header>
    {disabledReason ? <div className="mouse-protocol-notice" role="status"><p>{running ? "主链运行中，可以查看配置；停止后可修改、测试或切换协议。" : disabledReason}</p>{running && onGoHome ? <button type="button" className="console-button" onClick={onGoHome}>前往首页</button> : null}</div> : null}
    <form noValidate onSubmit={(event) => { event.preventDefault(); void submit(); }}>
      {protocol ? <label>协议<select aria-label="配置的协议" value={protocol.id} disabled={busy || draftChanged} onChange={(event) => onPick(event.target.value)}>
        {protocols.map((item) => <option key={item.id} value={item.id}>{models[item.model].label} · {item.host}:{item.port}{item.id === protocol.id && selected ? " · 已启用" : ""}</option>)}</select>
        {draftChanged ? <small className="mouse-protocol-test-hint">当前有未保存修改，请先保存或取消，再切换要配置的协议。</small> : null}</label> : null}
      <label>型号<select value={draft.model} disabled={busy || disabled} onChange={(event) => updateDraft({ model: event.target.value as MouseProtocol["model"] })}>
        {Object.entries(models).map(([value, model]) => <option key={value} value={value}>{model.label}</option>)}</select></label>
      <label>IP<input aria-label="IP" data-protocol-field="host" aria-invalid={invalidField === "host" || undefined} aria-describedby={invalidField === "host" ? errorId : undefined} type="text" inputMode="decimal" autoComplete="off" placeholder={defaults.host} value={draft.host} disabled={busy || disabled} onChange={(event) => updateDraft({ host: event.target.value })} />{fieldError("host")}</label>
      <label>Port<input aria-label="Port" data-protocol-field="port" aria-invalid={invalidField === "port" || undefined} aria-describedby={invalidField === "port" ? errorId : undefined} type="number" inputMode="numeric" min={1} max={65535} placeholder={String(defaults.port)} value={Number.isNaN(draft.port) ? "" : draft.port} disabled={busy || disabled} onChange={(event) => updateDraft({ port: event.target.value === "" ? NaN : Number(event.target.value) })} />{fieldError("port")}</label>
      <label>UUID<input aria-label="UUID" data-protocol-field="uuid" aria-invalid={invalidField === "uuid" || undefined} aria-describedby={invalidField === "uuid" ? errorId : undefined} type="text" autoComplete="off" maxLength={8} placeholder={defaults.uuid} value={draft.uuid} disabled={busy || disabled} onChange={(event) => updateDraft({ uuid: event.target.value })} />{fieldError("uuid")}</label>
      {error && invalidField === null ? <p className="action-confirmation-error" role="alert">{error}</p> : null}
      {result !== null ? <div className="mouse-protocol-result success" role="status"><NovaIcon name="check-circle" size={18} />设备握手成功 · {result} ms · 测试会话已断开</div> : null}
      <p className="mouse-protocol-test-hint">{previewOnly ? "预览模式不连接真实设备。" : "测试只握手并断开，不保存配置、不启用协议，也不发送位移。"}</p>
      {protocol && onRemove ? <div className="mouse-protocol-remove">
        {confirmRemove ? <div role="group" aria-label="确认删除协议"><p>删除此协议及已保存的连接参数？此操作不可撤销。</p><button type="button" className="console-button" disabled={busy} onClick={() => setConfirmRemove(false)}>保留协议</button><button type="button" className="console-button danger" disabled={busy || disabled || selected} onClick={() => void perform("remove", async () => { await onRemove(protocol); onClose(); })}>{operation === "remove" ? "正在删除…" : "确认删除"}</button></div> : <button type="button" className="console-button danger" disabled={busy || disabled || selected} onClick={() => setConfirmRemove(true)}>删除协议</button>}
        {selected ? <small>当前协议已启用，请选择空卡片取消启用，再删除。</small> : null}
      </div> : null}
      <footer><button type="button" className="console-button mouse-protocol-probe" disabled={busy || !canProbe} onClick={() => void testConnection()}>{operation === "probe" ? "正在测试…" : "测试连通性"}</button>
        <button type="button" className="console-button" disabled={busy} onClick={onClose}>{running ? "关闭" : "取消"}</button>
        {!running ? <button type="submit" className="console-button primary" disabled={busy || disabled}>{operation === "save" ? "正在保存…" : protocol ? previewOnly ? "保存协议（预览）" : "保存并写入文件" : "添加协议"}</button> : null}</footer>
    </form>
    {protocol ? diagnostics : null}
  </section></div>;
}
