import { NovaIcon, type NovaIconName } from "../../components/visual";
import type {
  ProductConfigAction,
  ProductConfigItem,
  ProductConfigItemId,
  ProductConfigProfile,
  ProductConfigState
} from "./productConfigProfile";

const ITEM_ICONS: Record<ProductConfigItemId, NovaIconName> = {
  capture: "capture",
  model: "models",
  control: "control",
  output: "device-send",
  kmnet: "hid",
  reload: "restart"
};

const STATE_ICONS: Record<ProductConfigState, NovaIconName> = {
  live: "check-circle",
  saved: "save",
  restart: "restart",
  missing: "triangle-alert",
  paused: "pause-output"
};

const STATE_LABELS: Record<ProductConfigState, string> = {
  live: "已生效",
  saved: "已保存",
  restart: "待重启",
  missing: "缺失",
  paused: "暂停"
};

const TONE_CLASSES: Record<ProductConfigState, "ready" | "waiting" | "blocked"> = {
  live: "ready",
  saved: "waiting",
  restart: "blocked",
  missing: "blocked",
  paused: "waiting"
};

function ProductConfigItemCard({
  item,
  busy,
  onAction
}: {
  item: ProductConfigItem;
  busy: boolean;
  onAction: (action: ProductConfigAction) => void;
}) {
  const tone = TONE_CLASSES[item.state];
  return (
    <li className={`control-trace-step product-config-profile-item ${tone} ${item.state}`}>
      <span className="control-trace-step-icon product-config-profile-item-icon" aria-hidden="true">
        <NovaIcon name={ITEM_ICONS[item.id]} size={18} strokeWidth={1.9} />
      </span>
      <div className="control-trace-step-copy">
        <div className="control-trace-step-heading product-config-profile-item-heading">
          <strong>{item.label}</strong>
          <span className={`control-trace-state product-config-profile-state ${tone} ${item.state}`}>
            <NovaIcon name={STATE_ICONS[item.state]} size={12} />
            {STATE_LABELS[item.state]}
          </span>
        </div>
        <b>{item.value}</b>
        <p>{item.detail}</p>
        <small>{item.evidence}</small>
      </div>
      {item.action ? (
        <button
          className="product-config-profile-action"
          disabled={busy}
          onClick={() => onAction(item.action as ProductConfigAction)}
          type="button"
        >
          {item.actionLabel}
        </button>
      ) : null}
    </li>
  );
}

export function ProductConfigProfilePanel({
  profile,
  busy,
  onAction
}: {
  profile: ProductConfigProfile;
  busy: boolean;
  onAction: (action: ProductConfigAction) => void;
}) {
  const tone = TONE_CLASSES[profile.state];
  return (
    <section className={`control-trace-panel product-config-profile-panel ${tone} ${profile.state}`} aria-labelledby="product-config-profile-title">
      <header className="control-trace-header product-config-profile-header">
        <div>
          <span className="class-config-eyebrow">配置摘要</span>
          <h2 id="product-config-profile-title">{profile.title}</h2>
          <p>{profile.detail}</p>
        </div>
        <div className="control-trace-count product-config-profile-score" aria-label={`配置生效 ${profile.liveCount}/${profile.totalCount}`}>
          <strong>{profile.liveCount}</strong>
          <span>/ {profile.totalCount}</span>
          <small>{profile.attentionCount > 0 ? `${profile.attentionCount} 项需处理` : "配置项"}</small>
        </div>
      </header>

      <ol className="control-trace-steps product-config-profile-items" aria-label="配置检查项">
        {profile.items.map((item) => (
          <ProductConfigItemCard
            key={item.id}
            item={item}
            busy={busy}
            onAction={onAction}
          />
        ))}
      </ol>

      <div className="control-trace-facts product-config-profile-facts" aria-label="配置摘要">
        {profile.facts.map((fact) => (
          <div key={fact.label}>
            <span>{fact.label}</span>
            <strong>{fact.value}</strong>
            <small>{fact.detail}</small>
          </div>
        ))}
      </div>
    </section>
  );
}
