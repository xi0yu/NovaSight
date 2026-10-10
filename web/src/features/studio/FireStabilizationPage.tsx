import { useEffect, useState } from "react";
import { ModuleSwitch } from "./ControlSwitches";
import { LoadedImage } from "../../components/ui";
import { NovaIcon } from "../../components/visual";
import { ParameterNumberControl } from "./StudioControls";
import sceneArt from "../../assets/themes/nova-daylight.webp";

import "./fire-stabilization.css";

export function FireStabilizationPage({ enabled, appliedEnabled = enabled, strength, runtimeStatus = "运行状态待核实", onEnabled, onStrength }: {
  enabled: boolean;
  appliedEnabled?: boolean;
  strength: number;
  runtimeStatus?: string;
  onEnabled: (enabled: boolean) => Promise<void> | void;
  onStrength: (strength: number) => Promise<void> | void;
}) {
  const [draftStrength, setDraftStrength] = useState<number | null>(null);
  useEffect(() => setDraftStrength(null), [strength]);
  const shownStrength = draftStrength ?? strength;
  const points = Array.from({ length: 11 }, (_, index) => {
    const lift = index * 12 + (index % 2 ? 8 : 0);
    const correction = enabled ? shownStrength * 0.65 * Math.max(0, (index - 2) / 8) : 0;
    return { x: 28 + index * 44, rawY: 174 - lift, adjustedY: 174 - lift * (1 - correction) };
  });
  const path = (adjusted: boolean) => points.map((point, index) => `${index === 0 ? "M" : "L"}${point.x},${adjusted ? point.adjustedY : point.rawY}`).join(" ");
  return <div className="fire-stability-page">
    <section className="fire-stability-hero" aria-labelledby="fire-stability-title">
      <LoadedImage className="fire-stability-scene" src={sceneArt} alt="" width="1536" height="1024" aria-hidden="true" />
      <div className="fire-stability-hero-copy">
        <span className="fire-stability-kicker"><NovaIcon name="shield-check" size={16} />自适应稳定辅助</span>
        <h2 id="fire-stability-title">让每次开火，更从容</h2>
        <p>接近目标时，逐渐抵消画面的上抬。<br />把注意力留给目标，把细微调整交给反馈。</p>
        <span className="fire-stability-runtime"><NovaIcon name="activity-pulse" size={14} />{runtimeStatus}</span>
      </div>
    </section>

    <div className="fire-stability-workbench">
      <section className="fire-stability-settings" aria-label="开火稳定设置">
        <header><span>你的稳定手感</span><h3>由你掌握介入力度</h3></header>
        <div className="fire-stability-switch-row">
          <span className="fire-stability-spark" aria-hidden="true"><NovaIcon name="target" size={20} /></span>
          <ModuleSwitch label="开火时稳定" detail={enabled !== appliedEnabled ? `等待自动保存 · ${enabled ? "开启" : "关闭"}` : enabled ? "已开启" : "已关闭"} enabled={enabled} onToggle={onEnabled} />
        </div>
        <ParameterNumberControl label="抑制强度" detail="越高，向下补偿越积极；0 为关闭补偿。" value={strength} min={0} max={1} step={0.05} kind="slider" applyMode="auto" riskLevel="normal" onDraftChange={setDraftStrength} onCommit={(value) => { setDraftStrength(null); return onStrength(value); }} />
        <p className="fire-stability-setting-note"><NovaIcon name="shield-check" size={16} />开启辅助不会启动设备。松开左键后回到原始跟随。</p>
      </section>

      <section className="fire-stability-preview" aria-labelledby="fire-effect-title">
        <header><div><span>看懂它如何帮你</span><h3 id="fire-effect-title">上抬抑制示意</h3></div><b>{enabled ? `强度 ${Math.round(shownStrength * 100)}%` : "辅助关闭"}</b></header>
        <div className="fire-stability-legend"><span className="raw">未经补偿</span><span className="adjusted">当前选择</span></div>
        <svg viewBox="0 0 520 210" role="img" aria-label={`上抬抑制机制示意，${enabled ? `强度 ${Math.round(shownStrength * 100)}%` : "辅助关闭"}，非设备实测`}>
          {[54, 114, 174].map((y) => <path key={y} d={`M28 ${y}H490`} className="fire-preview-grid" />)}
          <path d="M28 24V174H490" className="fire-preview-axis" />
          <path d={path(false)} className="fire-preview-raw" />
          <path d={path(true)} className="fire-preview-adjusted" />
          <circle cx={points[10].x} cy={points[10].adjustedY} r="5" className="fire-preview-dot" />
        </svg>
        <div className="fire-stability-chart-labels"><span>相对上抬</span><span>开火过程 →</span></div>
        <p>机制示意，非设备实测曲线。实际介入取决于目标、画面反馈与输出上限。</p>
      </section>
    </div>

    <section className="fire-stability-principles" aria-label="稳定辅助的工作方式">
      <article><NovaIcon name="target" size={22} /><h3>靠近目标，才介入</h3><p>按住左键且瞄点接近准星，才开始估计上抬。</p></article>
      <article><NovaIcon name="activity-pulse" size={22} /><h3>看反馈，渐进调整</h3><p>根据设备确认输出后的连续画面，逐渐增加向下补偿。</p></article>
      <article><NovaIcon name="shield-check" size={22} /><h3>懂得介入，也懂收手</h3><p>松开、换目标或离开近距即重置；补偿仍受 Y 轴上限约束。</p></article>
    </section>
    <details className="fire-stability-details"><summary>工作原理与使用边界</summary><p>按住左键且瞄点接近准星时，利用设备已确认输出后的连续画面估计上抬，逐渐增加 Y 轴下移量。</p><p>小幅向上回调会减弱补偿，明显越过目标时恢复原始回调。机制图用于理解设置，不能替代当前设备的实测效果。</p></details>
  </div>;
}
