import { useState, type CSSProperties, type PointerEvent } from "react";
import { ParameterNumberControl, InlineTextControl } from "./StudioControls";

// Stable class identity colors; IDs and participation text remain visible without color.
const CLASS_COLORS = ["#226d9a", "#9b601f", "#80559f", "#22796a", "#b13f68", "#76701f", "#226f84", "#984c85", "#547a2e", "#615ca3", "#a15239", "#287875", "#8a5297", "#4e6e9a", "#816a36", "#666078"];
export const classStyle = (id: number) => ({ "--class-color": CLASS_COLORS[id % CLASS_COLORS.length] }) as CSSProperties;

export function parseClassValues(value: unknown): Record<string, number> {
  if (typeof value !== "string") return {};
  return Object.fromEntries(value.split(",").flatMap((entry) => {
    const [id, raw] = entry.trim().split(":");
    const n = Number(raw);
    return /^\d+$/.test(id) && raw?.trim() && Number.isFinite(n) && n >= 0 && n <= 1 ? [[String(Number(id)), n]] : [];
  }));
}

export function setClassValue(value: unknown, id: number, next: number): string {
  return Object.entries({ ...parseClassValues(value), [id]: next })
    .sort(([a], [b]) => Number(a) - Number(b)).map(([key, n]) => `${key}:${n}`).join(",");
}

type Props = {
  ids: number[];
  names: string[];
  selected: Set<number>;
  weights: Record<string, number>;
  xs: Record<string, number>;
  ys: Record<string, number>;
  defaultY: number;
  disabled?: boolean;
  onPoint: (id: number, x: number, y: number) => void;
  onWeight: (id: number, weight: number) => void;
  onToggle: (id: number) => void;
  onName: (id: number, name: string) => void | Promise<void>;
};

export function TargetClassEditor({ ids, names, selected, weights, xs, ys, defaultY, disabled, onPoint, onWeight, onToggle, onName }: Props) {
  const [active, setActive] = useState(ids[0] ?? 0);
  const [newId, setNewId] = useState("");
  const id = ids.includes(active) ? active : ids[0] ?? 0;
  const x = xs[id] ?? 0.5;
  const y = ys[id] ?? defaultY;
  const point = (event: PointerEvent<HTMLButtonElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = (n: number) => Math.round(Math.max(0, Math.min(1, n)) * 100) / 100;
    onPoint(id, ratio((event.clientX - rect.left) / rect.width), ratio((event.clientY - rect.top) / rect.height));
  };
  return <section className="target-class-editor" aria-label="类别选点与权重" style={classStyle(id)}>
    <div className="class-point-workbench">
      <header><h3>cls {id} · 框内选点</h3><span>X {Math.round(x * 100)}% / Y {Math.round(y * 100)}%</span></header>
      <button type="button" className="class-point-pad" aria-label={`cls ${id} 瞄点平面，方向键移动，Home 居中`} disabled={disabled}
        onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); point(event); }}
        onPointerMove={(event) => { if (event.currentTarget.hasPointerCapture(event.pointerId)) point(event); }}
        onPointerUp={(event) => { if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}
        onKeyDown={(event) => {
          const step = event.shiftKey ? 0.1 : 0.01;
          const clamp = (n: number) => Math.round(Math.max(0, Math.min(1, n)) * 100) / 100;
          if (event.key === "Home") { event.preventDefault(); onPoint(id, 0.5, 0.5); }
          if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
            event.preventDefault(); onPoint(id, clamp(x + (event.key === "ArrowRight" ? step : event.key === "ArrowLeft" ? -step : 0)), clamp(y + (event.key === "ArrowDown" ? step : event.key === "ArrowUp" ? -step : 0)));
          }
        }}>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d="M25 0V100 M50 0V100 M75 0V100 M0 25H100 M0 50H100 M0 75H100" /></svg>
        <span className="class-point-marker" style={{ left: `${x * 100}%`, top: `${y * 100}%` }}>＋</span>
        <span className="class-point-origin">0,0</span><span className="class-point-end">100,100</span>
      </button>
      <p>点击或拖动选点；比例相对于该类别检测框，不假定 cls 代表头或身体。</p>
      <div className="class-point-presets">{[["上部", 0.5, 0.22], ["中心", 0.5, 0.5], ["下部", 0.5, 0.75]].map(([label, px, py]) => <button type="button" disabled={disabled} key={label} onClick={() => onPoint(id, Number(px), Number(py))}>{label}</button>)}</div>
      <div className="class-point-numbers">
        <ParameterNumberControl label={`cls ${id} 水平位置`} value={Math.round(x * 100)} min={0} max={100} step={1} unit="%" applyMode="save" disabled={disabled} onCommit={(n) => onPoint(id, n / 100, y)} />
        <ParameterNumberControl label={`cls ${id} 垂直位置`} value={Math.round(y * 100)} min={0} max={100} step={1} unit="%" applyMode="save" disabled={disabled} onCommit={(n) => onPoint(id, x, n / 100)} />
      </div>
    </div>
    <div className="class-weight-workbench">
      <header><h3>候选类别</h3><span>远处偏好 0—1 · 越大越优先</span></header>
      <p>先选择参与的类别和框内瞄点。附近瞄点优先；都较远时，偏好才参与评分。</p>
      <div className="class-selector-grid" aria-label="类别选择">{ids.map((classId) => <button type="button" key={classId} style={classStyle(classId)} aria-label={`cls ${classId}`} aria-pressed={id === classId} disabled={disabled} onClick={() => setActive(classId)}>
        <span><i aria-hidden="true" />cls {classId}</span>
        <small>{selected.has(classId) ? `偏好 ${(weights[classId] ?? 0).toFixed(2)}` : "未参与"}</small>
      </button>)}</div>
      <div className="class-active-settings" key={id}>
        <div className="class-weight-row"><strong>cls {id}</strong>
          <InlineTextControl ariaLabel={`cls ${id} 名称`} value={names[id] ?? ""} placeholder="类别备注" onCommit={(name) => onName(id, name)} />
          <label><input type="checkbox" aria-label={`cls ${id} 参与目标选择`} checked={selected.has(id)} disabled={disabled} onChange={() => onToggle(id)} />参与</label></div>
        <ParameterNumberControl label={`cls ${id} 远处偏好`} value={weights[id] ?? 0} min={0} max={1} step={0.05} kind="slider" applyMode="save" disabled={disabled} onCommit={(n) => onWeight(id, n)} />
        <small>0 仍可被选中；禁止选择请关闭“参与”。偏好不改变移动力度。</small>
      </div>
      <form className="class-add" onSubmit={(event) => { event.preventDefault(); const n = Number(newId); if (/^\d+$/.test(newId) && n >= 0 && n <= 255 && !ids.includes(n)) { onWeight(n, 0.5); setActive(n); setNewId(""); } }}>
        <label>添加 cls<input aria-label="新类别 ID" type="number" min={0} max={255} step={1} value={newId} onChange={(event) => setNewId(event.target.value)} /></label>
        <button type="submit" disabled={disabled || !/^\d+$/.test(newId) || Number(newId) > 255 || ids.includes(Number(newId))}>添加</button>
      </form>
      <p className="class-score-note">选中的瞄点仍有效时保持，不因另一个类别偏好更高而跳转。按键触发时，松开再按可重新选择；当前瞄点失效时，其他候选仍需通过连续确认。</p>
    </div>
  </section>;
}
