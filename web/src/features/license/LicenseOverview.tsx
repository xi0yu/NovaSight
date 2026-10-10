import { useEffect, useState } from "react";
import { type LicenseStatus } from "../../api";
import { StatusIndicator, LoadedImage } from "../../components/ui";
import { NovaIcon, type NovaIconName } from "../../components/visual";
import type { LicenseFeature } from "../../contracts/license";
import { formatEpoch } from "../shared/format";
import { formatLicenseCategory } from "./format";
import "./license-benefits.css";
import companionArt from "../../assets/themes/nova-chibi.webp";

const FEATURE_GROUPS: { title: string; icon: NovaIconName; features: LicenseFeature[] }[] = [
  { title: "看清每一帧", icon: "capture", features: ["capture", "runtime"] },
  { title: "拓展你的视觉能力", icon: "ai-model", features: ["models", "tensorrt", "plugins"] },
  { title: "打造你的工作台", icon: "device-send", features: ["hardware_control", "config_read", "config_write"] },
];

const FEATURE_BENEFITS: Record<LicenseFeature, string> = {
  capture: "连接你的画面来源", runtime: "掌握启动与停止",
  models: "整理与选择识别模型", tensorrt: "使用 TensorRT 加速", plugins: "使用插件扩展",
  hardware_control: "控制已连接的输出设备", config_read: "查看已保存设置", config_write: "保存工作台设置",
};

export function LicenseOverview({ license }: {
  license: LicenseStatus | null;
}) {
  const [now, setNow] = useState(Date.now);
  const expiresAt = license?.expires_at ? license.expires_at * 1000 : null;
  const remaining = expiresAt === null ? null : Math.max(0, Math.ceil((expiresAt - now) / 1000));
  useEffect(() => {
    if (expiresAt === null) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [expiresAt]);
  const licenseValid = license?.valid === true;
  const features = licenseValid ? license.features : [];
  return <div className="license-overview">
      <section className="license-summary" aria-label="设备授权摘要">
        <header>
          <div className="license-access-copy"><span>NovaSight · 专属授权</span><h3>{licenseValid ? "为你解锁，属于你的掌控力" : "你的视觉工作台，待你开启"}</h3><p>{licenseValid ? `当前已授权 ${features.length} 项能力` : "授权确认后，在这里查看你的专属能力"}</p></div>
          <LoadedImage className="license-access-character" src={companionArt} width="96" height="96" alt="" aria-hidden="true" />
        </header>
        <div className="license-summary-status">
          <div className="license-summary-title"><NovaIcon name="shield-check" size={20} /><strong>设备授权</strong></div>
          <StatusIndicator tone={licenseValid ? "good" : "idle"}>
            {licenseValid ? "授权有效" : license ? "尚未授权" : "授权待核实"}
          </StatusIndicator>
        </div>
        <dl className="license-device-expiry" aria-label="设备授权状态">
          <div className="license-deadline"><dt>到期时间</dt><dd>{license?.expires_at ? formatEpoch(license.expires_at) : "尚未确认"}<small className="license-expiry-countdown">{remaining === null ? "到期时间待核实" : remaining === 0 ? (licenseValid ? "正在核验到期状态" : "授权不可用") : `倒计时 ${String(Math.floor(remaining / 3600)).padStart(2, "0")}:${String(Math.floor(remaining / 60) % 60).padStart(2, "0")}:${String(remaining % 60).padStart(2, "0")}`}</small></dd></div>
          <div><dt>设备模型 ID</dt><dd>{license?.device_model || "设备型号待核实"}</dd></div>
          <div><dt>授权类别</dt><dd><span className="license-category">{formatLicenseCategory(license)}</span></dd></div>
        </dl>
      </section>
      <section className="license-scope license-benefits" aria-labelledby="license-scope-title">
        <header className="license-benefits-heading"><div><span>EXCLUSIVE ACCESS</span><h3 id="license-scope-title">你的专属能力</h3></div>{features.length > 0 ? <span className="license-benefits-seal"><NovaIcon name="shield-check" size={16} />专属权益 · 已解锁</span> : null}</header>
        {features.length > 0 ? <div className="license-scope-grid">
          {FEATURE_GROUPS.map((group) => {
            const enabled = group.features.filter((feature) => features.includes(feature));
            return enabled.length > 0 ? <article className="license-benefit-card" key={group.title}>
              <header><span className="license-benefit-icon" aria-hidden="true"><NovaIcon name={group.icon} size={24} /></span><div><small>专属能力</small><h4>{group.title}</h4></div><span className="license-benefit-count">{enabled.length} 项</span></header>
              <ul>{enabled.map((feature) => <li key={feature}><NovaIcon name="check-circle" size={16} /><div><strong>{formatFeature(feature)}</strong><small>{FEATURE_BENEFITS[feature]}</small></div></li>)}</ul>
            </article> : null;
          })}
        </div> : <p className="license-scope-empty">{licenseValid ? "此授权未包含可用功能。" : license ? "授权生效后，可在这里查看可用功能。" : "正在核实授权范围。"}</p>}
      </section>
  </div>;
}

function formatFeature(feature: string): string {
  const labels: Record<string, string> = {
    capture: "画面采集",
    runtime: "运行控制",
    models: "模型仓库",
    plugins: "插件扩展",
    tensorrt: "TensorRT 加速",
    hardware_control: "硬件控制",
    config_read: "读取配置",
    config_write: "写入配置"
  };
  return labels[feature] ?? feature;
}
