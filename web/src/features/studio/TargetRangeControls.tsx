import { ParameterNumberControl } from "./StudioControls";
import { NovaIcon } from "../../components/visual";

export function TargetRangeControls({ radius, onRadiusCommit, onEditingChange }: {
  radius: number;
  onRadiusCommit: (radius: number) => void | Promise<void>;
  onEditingChange?: (editing: boolean) => void;
}) {
  return (
    <section className="search-range-workbench" aria-label="搜索范围调校">
      <div className="search-range-tuning">
        <div className="search-range-tuning-heading"><span className="algorithm-setting-icon" aria-hidden="true"><NovaIcon name="target" size={20} /></span><div><h2>搜索范围</h2><p>以准星为圆心，选择圆内的目标瞄点。</p></div></div>
        <ParameterNumberControl
          label="搜索半径" value={radius} min={0} max={1000} step={10} unit="px" applyMode="auto"
          onCommit={onRadiusCommit} onEditingChange={onEditingChange}
        />
        <div className="search-range-presets" role="group" aria-label="搜索半径快捷调整">
          {[{ value: 80, label: "近距离" }, { value: 180, label: "中距离" }, { value: 320, label: "大范围" }].map(({ value, label }) => <button type="button" key={value} aria-label={`${label}，${value} px`} aria-pressed={radius === value} onClick={() => void onRadiusCommit(value)}><span>{label}</span><b>{value}<small> px</small></b></button>)}
        </div>
      </div>
    </section>
  );
}
