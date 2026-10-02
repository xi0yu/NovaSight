import { ModuleSwitch } from "./ControlSwitches";
import { ParameterNumberControl } from "./StudioControls";
import chibiArt from "../../assets/themes/nova-chibi.webp";

import "./fire-stabilization.css";

export function FireStabilizationPage({ enabled, strength, onEnabled, onStrength }: {
  enabled: boolean;
  strength: number;
  onEnabled: (enabled: boolean) => Promise<void> | void;
  onStrength: (strength: number) => Promise<void> | void;
}) {
  return <div className="fire-stability-page">
    <section className="fire-stability-hero" aria-labelledby="fire-stability-title">
      <div>
        <span className="fire-stability-kicker">FIRE STABILITY · 开火辅助</span>
        <h2 id="fire-stability-title">接近目标后，稳住开火时的上跳</h2>
        <p>正常跟随不变；只有按住左键开火、目标已进入近距离，才根据连续画面微调下移量。</p>
      </div>
      <img src={chibiArt} alt="" width="112" height="124" loading="lazy" />
    </section>

    <section className="fire-stability-settings" aria-label="开火稳定设置">
      <div className="fire-stability-switch-row">
        <span className="fire-stability-spark" aria-hidden="true">✦</span>
        <ModuleSwitch
          label="开火时稳定"
          detail={enabled ? "已开启 · 仅在符合条件时介入" : "已关闭 · 保持原有瞄准输出"}
          enabled={enabled}
          onToggle={onEnabled}
        />
      </div>
      <ParameterNumberControl
        label="抑制强度"
        detail="调高会更积极地抵消持续上跳；初次建议从中间值开始。"
        value={strength}
        min={0}
        max={1}
        step={0.05}
        kind="slider"
        applyMode="save"
        riskLevel="normal"
        onCommit={onStrength}
      />
      <p className="fire-stability-boundary">设为 0 时不增加下移量。输出仍受现有设备限幅保护；此设置不改变目标选择和预测。</p>
    </section>

    <section className="fire-stability-map" aria-labelledby="fire-stability-map-title">
      <div className="fire-stability-map-head"><span>生效范围</span><h3 id="fire-stability-map-title">什么时候会介入？</h3></div>
      <div className="fire-stability-scenarios">
        <div><span>01 · 距离较远</span><b>保持原有跟随</b><small>无论向上还是向下，都不额外修正。</small></div>
        <div><span>02 · 已接近目标</span><b>连续上跳才补偿</b><small>多帧确认后，逐渐增加向下修正。</small></div>
        <div><span>03 · 需要向上回调</span><b>减少反复拉扯</b><small>仅在仍有补偿趋势时减弱小幅回调。</small></div>
        <div><span>04 · 松开或换目标</span><b>立即清空记忆</b><small>下一次开火重新观察，不继承旧目标的修正。</small></div>
      </div>
    </section>
  </div>;
}
