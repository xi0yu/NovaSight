import { useState } from "react";
import { ParameterNumberControl } from "./StudioControls";
import { NovaIcon } from "../../components/visual";

type Props = {
  radius: number;
  onRadiusCommit: (radius: number) => void | Promise<void>;
  onEditingChange?: (editing: boolean) => void;
};

const SAMPLES = [
  { label: "靠近准星", x: 420, y: 320 },
  { label: "中等距离", x: 540, y: 320 },
  { label: "远离准星", x: 600, y: 320 },
];

export function TargetRangeControls({ radius, onRadiusCommit, onEditingChange }: Props) {
  const [sample, setSample] = useState(1);
  const [draft, setDraft] = useState<number | null>(null);
  const shownRadius = draft ?? radius;
  const point = SAMPLES[sample];
  const distance = Math.hypot(point.x - 320, point.y - 320);
  const inside = distance <= shownRadius;

  return (
    <section className="search-range-workbench" aria-label="搜索范围调校">
      <div className="search-range-tuning">
        <div className="search-range-tuning-heading"><span className="algorithm-setting-icon" aria-hidden="true"><NovaIcon name="target" size={20} /></span><div><h2>搜索范围</h2><p>以准星为圆心，选择圆内的目标瞄点。</p></div></div>
        <ParameterNumberControl
          label="搜索半径"
          detail="按观测画面像素计算；目标大小不改变范围。"
          value={radius} min={0.000001} max={100000} recommendedMin={10} recommendedMax={500} step={10} unit="px" applyMode="save"
          onDraftChange={setDraft}
          onCommit={async (value) => { try { await onRadiusCommit(value); } finally { setDraft(null); } }}
          onEditingChange={(editing) => { if (!editing) setDraft(null); onEditingChange?.(editing); }}
        />
        <div className="search-range-presets" role="group" aria-label="搜索半径快捷调整">
          {[{ value: 80, label: "近距离" }, { value: 180, label: "中距离" }, { value: 320, label: "大范围" }].map(({ value, label }) => <button type="button" key={value} aria-label={`${value} px`} aria-pressed={shownRadius === value} onClick={() => { setDraft(null); void onRadiusCommit(value); }}><span>{label}</span><b>{value}<small> px</small></b></button>)}
        </div>
        <dl className="search-range-explainer"><div><dt>范围只负责筛选</dt><dd>圆内可以参与选择；圆外不输出。</dd></div><div><dt>力度独立调整</dt><dd>扩大范围不会增加跟随力度。</dd></div></dl>
        <details className="search-range-geometry"><summary>了解范围判定</summary><p>以最新检测框和类别选点计算瞄准点，点在圆内或圆周上才满足范围条件，不按检测框是否碰到圆来判断，也不拿预测位置放宽准入。范围内仍需满足触发和安全条件；范围外可以保留跟踪身份，但不允许输出。这里的“搜索”是推理后的候选筛选，不裁剪采集或推理画面。</p></details>
      </div>
      <div className="search-range-scene">
        <div className="search-range-scene-heading"><span><i aria-hidden="true" />范围预览</span><span>640 × 640 · 示意</span></div>
        <svg viewBox="0 0 640 640" role="img" aria-label={`搜索圆示意，半径 ${shownRadius} px，瞄准点${inside ? "在范围内" : "在范围外"}`}>
          <path className="search-range-grid" d="M0 160H640 M0 320H640 M0 480H640 M160 0V640 M320 0V640 M480 0V640" />
          <circle className="search-range-guide" cx="320" cy="320" r="280" />
          <circle className="search-range-guide" cx="320" cy="320" r="100" />
          <circle className="search-range-region" cx="320" cy="320" r={shownRadius} />
          <path className="search-range-distance" d={`M320 320H${point.x}`} />
          <rect className="search-range-detection" x={point.x - 20} y={point.y - 24} width="40" height="80" rx="3" />
          <text className="search-range-coordinate" x="320" y="358" textAnchor="middle">准星</text>
          <text className="search-range-coordinate" x={point.x} y="284" textAnchor="middle">目标</text>
          <path className="search-range-crosshair" d="M308 320h24 M320 308v24" />
          <circle className={`search-range-aim ${inside ? "inside" : "outside"}`} cx={point.x} cy={point.y} r="6" />
        </svg>
        <div className="search-range-measurements"><div><span>搜索半径</span><b>{shownRadius}<small>px</small></b></div><div><span>目标距离</span><b>{distance}<small>px</small></b></div></div>
        <div className="search-range-legend"><span><i />搜索范围</span><span><i />检测框</span><span>● 瞄准点</span><span>＋ 准星</span></div>
        <div className="search-range-samples" role="group" aria-label="模拟目标位置">
          {SAMPLES.map((item, index) => <button key={item.label} type="button" aria-pressed={sample === index} onClick={() => setSample(index)}>{item.label}</button>)}
        </div>
        <p className={`search-range-verdict ${inside ? "inside" : "outside"}`} role="status"><i aria-hidden="true" />{inside ? "范围条件满足" : "范围外 · 不输出"}<span>仅演示范围判定，不代表实际运行状态</span></p>
      </div>
    </section>
  );
}
