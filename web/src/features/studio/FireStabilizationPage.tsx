import { ModuleSwitch } from "./ControlSwitches";
import { LoadedImage } from "../../components/ui";
import { ParameterNumberControl } from "./StudioControls";
import chibiArt from "../../assets/themes/nova-chibi.webp";

import "./fire-stabilization.css";

export function FireStabilizationPage({ enabled, appliedEnabled = enabled, strength, onEnabled, onStrength }: {
  enabled: boolean;
  appliedEnabled?: boolean;
  strength: number;
  onEnabled: (enabled: boolean) => Promise<void> | void;
  onStrength: (strength: number) => Promise<void> | void;
}) {
  return <div className="fire-stability-page">
    <section className="fire-stability-hero" aria-labelledby="fire-stability-title">
      <div>
        <span className="fire-stability-kicker">开火辅助</span>
        <h2 id="fire-stability-title">抑制开火时的上抬</h2>
      </div>
      <LoadedImage src={chibiArt} alt="" width="112" height="124" loading="lazy" />
    </section>

    <section className="fire-stability-settings" aria-label="开火稳定设置">
      <div className="fire-stability-switch-row">
        <span className="fire-stability-spark" aria-hidden="true">✦</span>
        <ModuleSwitch
          label="开火时稳定"
          detail={enabled !== appliedEnabled ? `等待自动保存 · ${enabled ? "开启" : "关闭"}` : enabled ? "已开启" : "已关闭"}
          enabled={enabled}
          onToggle={onEnabled}
        />
      </div>
      <ParameterNumberControl
        label="抑制强度"
        detail="越高，向下补偿越积极；0 为关闭补偿。"
        value={strength}
        min={0}
        max={1}
        step={0.05}
        kind="slider"
        applyMode="auto"
        riskLevel="normal"
        onCommit={onStrength}
      />
    </section>

    <details className="fire-stability-details">
      <summary>工作原理</summary>
      <p>按住左键且瞄点接近准星时，利用设备已确认输出后的连续画面估计上抬，逐渐增加 Y 轴下移量。</p>
      <p>松开、换目标或离开近距离会重置。小幅向上回调会减弱，明显越过目标时恢复原始回调；最终输出仍受 Y 轴上限限制。</p>
    </details>
  </div>;
}
