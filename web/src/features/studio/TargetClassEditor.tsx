import { type CSSProperties, type PointerEvent } from "react";
import { ParameterNumberControl, InlineTextControl } from "./StudioControls";

// IDs and participation remain readable without relying on color.
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

export type ClassSettings = {
  names: string[]; selected: Set<number>;
  weights: Record<string, number>; xs: Record<string, number>; ys: Record<string, number>;
  defaultY: number;
};

type Props = ClassSettings & {
  ids: number[]; disabled?: boolean; suggestedNames?: string[];
  onPoint: (id: number, x: number, y: number) => void;
  onWeight: (id: number, weight: number) => void;
  onToggle: (id: number) => void;
  onName: (id: number, name: string) => void | Promise<void>;
  saved: ClassSettings;
  onReset: (id: number) => void;
};

export function TargetClassEditor({ ids, names, suggestedNames, selected, weights, xs, ys, defaultY, disabled, onPoint, onWeight, onToggle, onName, saved, onReset }: Props) {
  const legacyIds = ids.filter((id) => id > 7 && (selected.has(id) || names[id]?.trim() || weights[id] !== undefined || xs[id] !== undefined || ys[id] !== undefined));
  const ratio = (n: number) => Math.round(Math.max(0, Math.min(1, n)) * 100) / 100;
  const point = (id: number, event: PointerEvent<HTMLButtonElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) onPoint(id, ratio((event.clientX - rect.left) / rect.width), ratio((event.clientY - rect.top) / rect.height));
  };
  return <section className="target-class-editor" aria-label="类别选点与权重">
    <p className="class-table-note">类别偏好越大，首次选择时越倾向该类别；不会抢走已锁定的目标。0 不代表禁用，取消“参与”才会排除。</p>
    {suggestedNames?.some((name) => name.trim()) ? <p className="class-table-note" role="note">名称输入框里的模型登记名只是参考。只有填写并保存后才会写入当前配置。</p> : null}
    {legacyIds.length > 0 && <p className="class-legacy-note" role="note">cls8 以上的参与状态或已有配置仍保留：{legacyIds.map((id) => `cls${id}`).join("、")}。本页仅编辑 cls0～7，不会自动删除或停用其他类别。</p>}
    <div className="class-table-scroll" role="region" aria-label="八个目标类别配置" tabIndex={0}>
      <table className="class-edit-table">
        <thead><tr><th scope="col">参与</th><th scope="col">类别</th><th scope="col">名称</th><th scope="col">类别偏好</th><th scope="col">框内瞄点</th></tr></thead>
        <tbody>{Array.from({ length: 8 }, (_, id) => {
          const x = xs[id] ?? 0.5;
          const y = ys[id] ?? defaultY;
          const dirty = (names[id] ?? "") !== (saved.names[id] ?? "") || selected.has(id) !== saved.selected.has(id)
            || (weights[id] ?? 0) !== (saved.weights[id] ?? 0) || x !== (saved.xs[id] ?? 0.5) || y !== (saved.ys[id] ?? saved.defaultY);
          const matchesPreset = (py: number) => Math.abs(x - 0.5) < 0.000001 && Math.abs(y - py) < 0.000001;
          const preset = [0.22, 0.5, 0.75].some(matchesPreset);
          return <tr key={id} style={classStyle(id)} data-enabled={selected.has(id)}>
            <td><input type="checkbox" aria-label={`cls ${id} 参与目标选择`} checked={selected.has(id)} disabled={disabled} onChange={() => onToggle(id)} /><small className="class-participation">{selected.has(id) ? "参与" : "不参与"}</small></td>
            <th scope="row"><span className="class-table-id">cls{id}</span>{dirty && <span className="class-row-dirty" role="status" aria-label={`cls ${id} 已修改`} title="已修改，尚未保存" />}</th>
            <td><fieldset disabled={disabled} className="class-name-field"><InlineTextControl ariaLabel={`cls ${id} 名称`} value={names[id] ?? ""} placeholder={suggestedNames?.[id]?.trim() ? `模型登记：${suggestedNames[id].trim()}` : "填写名称"} onCommit={(name) => onName(id, name)} /></fieldset></td>
            <td><ParameterNumberControl label={`cls ${id} 类别偏好`} value={weights[id] ?? 0} min={0} max={1} step={0.05} kind="slider" applyMode="save" disabled={disabled} onCommit={(n) => onWeight(id, n)} /></td>
            <td><div className="class-inline-point">
              <button type="button" className="class-point-pad" aria-label={`cls ${id} 瞄点平面，方向键移动，Home 居中`} disabled={disabled}
                onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); point(id, event); }}
                onPointerMove={(event) => { if (event.currentTarget.hasPointerCapture(event.pointerId)) point(id, event); }}
                onPointerUp={(event) => { if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}
                onKeyDown={(event) => {
                  const step = event.shiftKey ? 0.1 : 0.01;
                  if (event.key === "Home") { event.preventDefault(); onPoint(id, 0.5, 0.5); }
                  if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
                    event.preventDefault(); onPoint(id, ratio(x + (event.key === "ArrowRight" ? step : event.key === "ArrowLeft" ? -step : 0)), ratio(y + (event.key === "ArrowDown" ? step : event.key === "ArrowUp" ? -step : 0)));
                  }
                }}>
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d="M50 0V100 M0 50H100" /></svg>
                <span className="class-point-marker" style={{ left: `${x * 100}%`, top: `${y * 100}%` }}>＋</span>
              </button>
              <div><div className="class-point-presets">{[["上部", 0.22], ["中心", 0.5], ["下部", 0.75]].map(([label, py]) => <button type="button" disabled={disabled} aria-pressed={matchesPreset(Number(py))} aria-label={`cls ${id} ${label}`} key={label} onClick={() => onPoint(id, 0.5, Number(py))}>{label}</button>)}</div>
                <small>{!preset && "自定义 · "}水平 {Math.round(x * 100)}% · 垂直 {Math.round(y * 100)}%</small>
                <details className="class-point-precision"><summary>精确坐标<span className="sr-only"> cls{id}</span></summary><div className="class-point-numbers">
                  <ParameterNumberControl label={`cls ${id} 水平位置`} value={Math.round(x * 100)} min={0} max={100} step={1} unit="%" applyMode="save" disabled={disabled} onCommit={(n) => onPoint(id, n / 100, y)} />
                  <ParameterNumberControl label={`cls ${id} 垂直位置`} value={Math.round(y * 100)} min={0} max={100} step={1} unit="%" applyMode="save" disabled={disabled} onCommit={(n) => onPoint(id, x, n / 100)} />
                </div></details>
                {dirty && <button type="button" className="class-row-reset" aria-label={`撤销 cls ${id} 修改`} disabled={disabled} onClick={() => onReset(id)}>撤销本行</button>}
              </div>
            </div></td>
          </tr>;
        })}</tbody>
      </table>
    </div>
    <p className="class-table-note">点击小框设置瞄点，或选择上部、中心、下部。修改后回到参数页统一保存。</p>
  </section>;
}
