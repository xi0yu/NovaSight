import { useState } from "react";
import type { ClassPreset } from "../../contracts/config";
import { classStyle } from "./TargetClassEditor";

import "./class-preset-picker.css";

export function ClassPresetPicker({ presets, canSave, onApply, onSave }: {
  presets: ClassPreset[];
  canSave: boolean;
  onApply: (preset: ClassPreset) => void;
  onSave: (name: string) => void;
}) {
  const [selectedId, setSelectedId] = useState("");
  const [name, setName] = useState("");
  const selected = presets.find((preset) => preset.id === selectedId);
  return <section className="class-preset-picker" aria-label="类别预设">
    <div className="class-preset-intro"><span>CLASS MAP · 类别编排</span><b>先选一套，再按需微调</b><small>预设填写类别名称、参与状态和偏好；类别专属瞄点回到默认。不会自动切换模型。</small></div>
    <div className="class-preset-actions">
      <label><span className="sr-only">选择类别预设</span><select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>
        <option value="">选择预设…</option>
        {presets.map((preset) => <option value={preset.id} key={preset.id}>{preset.label}{preset.verified ? "" : " · 待核对"}</option>)}
      </select></label>
      <button className="console-button primary" type="button" disabled={!selected} onClick={() => selected && onApply(selected)}>应用到当前草稿</button>
    </div>
    {selected ? <div className="class-preset-preview" role="note">
      <p>{selected.note || "使用前核对模型输出类别。"}</p>
      <div>{selected.class_names.map((label, id) => <span key={id} style={classStyle(id)} data-enabled={selected.enabled_ids.includes(id)}><i />cls{id} {label}</span>)}</div>
    </div> : null}
    <div className="class-preset-save">
      <label><span>保存当前编排为预设</span><input maxLength={40} placeholder="例如：我的七类配置" value={name} onChange={(event) => setName(event.target.value)} /></label>
      <button className="console-button" type="button" disabled={!canSave || !name.trim()} onClick={() => { onSave(name.trim()); setName(""); }}>保存预设</button>
    </div>
    {!canSave ? <small className="class-preset-hint">先选择至少一个参与类别，再保存预设。</small> : null}
  </section>;
}
