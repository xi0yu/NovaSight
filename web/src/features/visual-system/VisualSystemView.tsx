import { Fragment } from "react";

import { EmptyStateVisual, NovaIcon, StatusBadge, ThemeToggle } from "../../components/visual";
import { iconCategories, type NovaIconName } from "../../design/iconNames";
import { visualAssetGroups } from "../../design/visualAssets";
import type { NovaStatus } from "../../design/statusTokens";

const svgAssets = import.meta.glob("../../assets/**/*.svg", {
  eager: true,
  import: "default",
  query: "?url",
}) as Record<string, string>;

const statusSamples: { status: NovaStatus; label: string; detail: string }[] = [
  { status: "normal", label: "TensorRT 已就绪", detail: "模型已加载" },
  { status: "running", label: "正在推理", detail: "latest-frame" },
  { status: "waiting", label: "等待输入", detail: "未启动" },
  { status: "warning", label: "延迟偏高", detail: "观察队列" },
  { status: "error", label: "设备断开", detail: "检查连接" },
  { status: "disabled", label: "当前不可用", detail: "权限不足" },
];

const emptySamples = [
  { icon: "devices", title: "尚未连接设备", detail: "连接采集设备后才能启动实时视觉链路。" },
  { icon: "models", title: "尚未选择模型", detail: "选择 TensorRT engine 或 ONNX 模型作为当前推理产物。" },
  { icon: "target", title: "暂无检测结果", detail: "采集和推理启动后会显示目标框、新鲜度和控制观察。" },
] satisfies { icon: NovaIconName; title: string; detail: string }[];

function visualAssetUrl(assetPath: string): string {
  return svgAssets[`../../assets/${assetPath}`] ?? "";
}

function AssetStrip({ title, assets }: { title: string; assets: readonly string[] }) {
  return (
    <section className="visual-section">
      <div className="visual-section-head">
        <h2>{title}</h2>
        <StatusBadge status="normal" label={`${assets.length} files`} size="sm" />
      </div>
      <div className="visual-asset-grid">
        {assets.map((asset) => (
          <figure className="visual-asset" key={asset}>
            <img alt="" src={visualAssetUrl(asset)} />
            <figcaption>{asset}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}

function IconCategory({ title, icons }: { title: string; icons: readonly NovaIconName[] }) {
  return (
    <section className="visual-section">
      <div className="visual-section-head">
        <h2>{title}</h2>
        <StatusBadge status="running" label={`${icons.length} icons`} size="sm" />
      </div>
      <div className="visual-icon-grid">
        {icons.map((icon) => (
          <div className="visual-icon-cell" key={icon}>
            <NovaIcon name={icon} size={22} />
            <span>{icon}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

export function VisualSystemView() {
  return (
    <main className="visual-system-page">
      <header className="visual-hero">
        <div className="visual-hero-brand">
          <span className="brand-mark">
            <NovaIcon name="prediction-line" size={24} strokeWidth={1.9} />
          </span>
          <div>
            <h1>NovaSight Visual System</h1>
            <p>SVG icons, status badges, brand marks, feature illustrations, and empty states.</p>
          </div>
        </div>
        <ThemeToggle />
      </header>

      <section className="visual-section">
        <div className="visual-section-head">
          <h2>Status Badges</h2>
          <StatusBadge status="normal" label="icon + color + text" size="sm" />
        </div>
        <div className="visual-status-grid">
          {statusSamples.map((sample) => (
            <StatusBadge
              detail={sample.detail}
              key={sample.status}
              label={sample.label}
              status={sample.status}
            />
          ))}
        </div>
      </section>

      <AssetStrip assets={visualAssetGroups.logos} title="Brand Logos" />
      <AssetStrip assets={visualAssetGroups.featureIllustrations} title="Feature Illustrations" />
      <AssetStrip assets={visualAssetGroups.emptyStates} title="Empty State Assets" />
      <AssetStrip assets={visualAssetGroups.exportedIcons} title="Exported SVG Icon Sources" />

      <section className="visual-section">
        <div className="visual-section-head">
          <h2>Empty State Components</h2>
          <StatusBadge status="normal" label="component anatomy" size="sm" />
        </div>
        <div className="visual-empty-grid">
          {emptySamples.map((sample) => (
            <EmptyStateVisual
              detail={sample.detail}
              icon={sample.icon}
              key={sample.title}
              title={sample.title}
            />
          ))}
        </div>
      </section>

      {Object.entries(iconCategories).map(([category, icons]) => (
        <Fragment key={category}>
          <IconCategory icons={icons} title={`Icons / ${category}`} />
        </Fragment>
      ))}
    </main>
  );
}
