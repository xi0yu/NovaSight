import { useMemo, useState, type ReactNode } from "react";

import { NovaIcon } from "../../components/visual";

import "./activity-view.css";

function formatActivityTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export type ActivityItem = {
  key: string;
  title: string;
  detail: string;
  technicalDetail?: string;
  requestId?: string | null;
  tag?: string;
  time?: number;
  count?: number;
  tone?: "info" | "warn" | "error" | "success";
};

type ActivityFilter = "all" | "error" | "warn" | "success" | "info";

const DEFAULT_MODULES = ["summary", "filters", "feed"];

export function ActivityView({
  items,
  onOpenDetails,
  moduleOrder = DEFAULT_MODULES,
}: {
  items: ActivityItem[];
  onOpenDetails: () => void;
  moduleOrder?: readonly string[];
}) {
  const [filter, setFilter] = useState<ActivityFilter>("all");
  const attentionCount = items.filter((item) => item.tone === "warn" || item.tone === "error" || item.tone === undefined).length;
  const completedCount = items.filter((item) => item.tone === "success").length;
  const filteredItems = useMemo(() => filter === "all"
    ? items
    : items.filter((item) => (item.tone ?? "error") === filter), [filter, items]);
  const counts = useMemo(() => ({
    all: items.length,
    error: items.filter((item) => (item.tone ?? "error") === "error").length,
    warn: items.filter((item) => item.tone === "warn").length,
    success: items.filter((item) => item.tone === "success").length,
    info: items.filter((item) => item.tone === "info").length,
  }), [items]);

  const modules: Record<string, ReactNode> = {
    summary: (
      <header className={attentionCount > 0 ? "activity-summary needs-attention" : "activity-summary is-clear"} aria-live="polite">
        <span className="activity-summary-icon" aria-hidden="true">
          <NovaIcon name={attentionCount > 0 ? "triangle-alert" : "shield-check"} size={22} />
        </span>
        <div>
          <span>现在</span>
          <h2 id="activity-view-title">{attentionCount > 0 ? `${attentionCount} 件事需要注意` : completedCount > 0 ? "最近操作已完成" : items.length > 0 ? "最近有新记录" : "暂无活动记录"}</h2>
          <p>{attentionCount > 0 ? "后端故障会在服务运行期间保留；处理建议优先展示，底层证据按需展开。" : "这里汇总后端运行事件和当前页面操作；运行结论请查看首页状态。"}</p>
        </div>
        {attentionCount > 0 ? (
          <button className="console-button secondary" onClick={onOpenDetails} type="button">查看完整详情</button>
        ) : null}
      </header>
    ),
    filters: (
      <nav className="activity-filters" aria-label="日志等级筛选">
        {([
          ["all", "全部"], ["error", "故障"], ["warn", "警告"], ["success", "完成"], ["info", "信息"],
        ] as const).map(([value, label]) => (
          <button
            aria-pressed={filter === value}
            data-tone={value}
            key={value}
            onClick={() => setFilter(value)}
            type="button"
          >
            <span>{label}</span><b>{counts[value]}</b>
          </button>
        ))}
      </nav>
    ),
    feed: (
      <div className="activity-feed" aria-label="最近活动">
        {filteredItems.length > 0 ? filteredItems.map((item) => (
          <article className="activity-feed-item" data-tone={item.tone ?? "error"} key={item.key}>
            <span className="activity-feed-marker" aria-hidden="true" />
            <div>
              <div className="activity-item-tags">
                <span className="activity-level-tag" data-tone={item.tone ?? "error"}>{item.tone === "success" ? "完成" : item.tone === "info" ? "信息" : item.tone === "warn" ? "警告" : "故障"}</span>
                {item.tag ? <span className="activity-domain-tag">{item.tag}</span> : null}
              </div>
              <strong>{item.title}</strong>
              {(item.count ?? 1) > 1 ? <small>本次会话重复 {item.count} 次</small> : null}
              <p>{item.detail}</p>
              {item.time ? <time>{formatActivityTime(item.time)}</time> : <time>仍在发生</time>}
              {item.technicalDetail ? (
                <details>
                  <summary>原始错误与开发者详情</summary>
                  <pre>{item.technicalDetail}</pre>
                </details>
              ) : null}
            </div>
          </article>
        )) : (
          <div className="activity-empty">
            <span>{items.length > 0 ? "这个等级暂时没有日志" : "暂无运行记录"}</span>
            <small>{items.length > 0 ? "选择其他等级继续查看。" : "运行、模型、采集或配置发生变化时，会按时间出现在这里。"}</small>
          </div>
        )}
      </div>
    ),
  };

  return (
    <section className="console-page activity-view" aria-labelledby="activity-view-title">
      {moduleOrder.map((moduleId) => <div data-module={moduleId} key={moduleId}>{modules[moduleId] ?? null}</div>)}
    </section>
  );
}
