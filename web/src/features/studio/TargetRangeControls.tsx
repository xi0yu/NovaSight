import { useState } from "react";
import { ParameterNumberControl } from "./StudioControls";

type Props = {
  scale: number;
  onScaleCommit: (scale: number) => void | Promise<void>;
  onEditingChange?: (editing: boolean) => void;
};

const SAMPLES = [
  { label: "贴近目标", x: 210, y: 130 },
  { label: "稍有偏离", x: 226, y: 130 },
  { label: "远离目标", x: 258, y: 130 },
];

export function TargetRangeControls({ scale, onScaleCommit, onEditingChange }: Props) {
  const [sample, setSample] = useState(1);
  const [draft, setDraft] = useState<number | null>(null);
  const shownScale = draft ?? scale;
  const point = SAMPLES[sample];
  const radius = 16 * shownScale;
  const halfSegment = 20 * shownScale;
  const dx = point.x - 200;
  const dy = Math.max(0, Math.abs(point.y - 130) - halfSegment);
  const inside = dx * dx + dy * dy <= radius * radius;

  return (
    <section className="capsule-workbench" aria-label="胶囊范围调校">
      <div className="capsule-scene">
        <div className="capsule-scene-heading"><span>范围示意</span><span>不连接设备</span></div>
        <svg viewBox="0 -65 400 390" role="img" aria-label={`胶囊范围示意，比例 ${Math.round(shownScale * 100)}%，准星${inside ? "在范围内" : "在范围外"}`}>
          <path className="capsule-grid" d="M0 65H400 M0 130H400 M0 195H400 M100 0V260 M200 0V260 M300 0V260" />
          <rect className="capsule-detection" x="184" y="94" width="32" height="72" />
          <path className="capsule-silhouette" d="M200 112v25m-10-15 10-6 10 6m-10 15-9 19m9-19 9 19" />
          <circle className="capsule-silhouette" cx="200" cy="103" r="5" />
          <rect className="capsule-region" x={200 - radius} y={130 - 36 * shownScale} width={32 * shownScale} height={72 * shownScale} rx={radius} />
          <path className={`capsule-crosshair ${inside ? "inside" : "outside"}`} d={`M${point.x - 8} ${point.y}h16 M${point.x} ${point.y - 8}v16`} />
        </svg>
        <div className="capsule-legend"><span><i />辅助范围</span><span><i />检测框</span><span>＋ 准星</span></div>
        <div className="capsule-samples" role="group" aria-label="模拟准星位置">
          {SAMPLES.map((item, index) => <button key={item.label} type="button" aria-pressed={sample === index} onClick={() => setSample(index)}>{item.label}</button>)}
        </div>
        <p className={`capsule-verdict ${inside ? "inside" : "outside"}`} role="status">{inside ? "范围条件满足" : "范围外 · 不输出"}<span>仅演示几何判定，不代表实际运行状态</span></p>
      </div>
      <div className="capsule-tuning">
        <div className="capsule-tuning-heading"><h2>靠近多少，开始辅助</h2><p>范围跟随目标大小。准星进入胶囊后，才继续判断触发条件和控制输出。</p></div>
        <ParameterNumberControl
          label="范围比例"
          detail="100% 为基础胶囊；150% 将宽高整体放大到 1.5 倍。"
          value={Math.round(scale * 100)} min={10} max={500} step={10} unit="%" applyMode="save"
          onDraftChange={(percent) => setDraft(percent / 100)}
          onCommit={async (percent) => { try { await onScaleCommit(percent / 100); } finally { setDraft(null); } }}
          onEditingChange={(editing) => { if (!editing) setDraft(null); onEditingChange?.(editing); }}
        />
        <div className="capsule-scale-presets" role="group" aria-label="范围比例快捷调整">
          {[100, 200, 500].map((percent) => <button type="button" key={percent} aria-pressed={Math.round(shownScale * 100) === percent} onClick={() => { setDraft(null); void onScaleCommit(percent / 100); }}>{percent}%</button>)}
        </div>
        <dl className="capsule-explainer"><div><dt>缩小范围</dt><dd>更靠近目标才介入，减少提前拉动。</dd></div><div><dt>放大范围</dt><dd>更早获得辅助，不会提高移动强度。</dd></div></dl>
        <details className="capsule-geometry"><summary>范围如何计算</summary><p>以最新检测框中心为中心，整体缩放竖直胶囊。宽矮框退化为圆，范围可能超过框的高度。范围外可以保留跟踪身份，但不允许输出。</p></details>
      </div>
    </section>
  );
}
