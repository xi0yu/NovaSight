import { useMemo, useState, type ReactNode } from "react";

import { NovaIcon } from "../../components/visual";
import { EmptyState } from "../../components/ui/EmptyState";
import { getErrorMessage } from "../shared/format";

import "./activity-view.css";

function formatActivityTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString("zh-CN", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
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
  onClear,
  onOpenDetails,
  moduleOrder = DEFAULT_MODULES,
}: {
  items: ActivityItem[];
  onClear: () => Promise<void> | void;
  onOpenDetails: () => void;
  moduleOrder?: readonly string[];
}) {
  const [filter, setFilter] = useState<ActivityFilter>("all");
  const [clearing, setClearing] = useState(false);
  const [query, setQuery] = useState("");
  const [clearError, setClearError] = useState("");
  const attentionCount = items.filter((item) => item.tone === "warn" || item.tone === "error" || item.tone === undefined).length;
  const completedCount = items.filter((item) => item.tone === "success").length;
  const filteredItems = useMemo(() => items.filter((item) =>
    (filter === "all" || (item.tone ?? "error") === filter)
    && [item.title, item.detail, item.tag, item.requestId, item.technicalDetail].join(" ").toLocaleLowerCase().includes(query.trim().toLocaleLowerCase())
  ), [filter, items, query]);
  const clearHistory = async () => {
    setClearing(true);
    setClearError("");
    try { await onClear(); }
    catch (error) { setClearError(getErrorMessage(error)); }
    finally { setClearing(false); }
  };
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
          <span>活动记录 · {items.length} 条</span>
          <h2 id="activity-view-title">{attentionCount > 0 ? `${attentionCount} 条故障与警告记录` : completedCount > 0 ? "最近操作已完成" : items.length > 0 ? "最近有新记录" : "暂无活动记录"}</h2>
          <p>记录不代表故障仍在发生；当前运行状态请查看首页。</p>
        </div>
        {attentionCount > 0 ? (
          <button className="console-button secondary" onClick={onOpenDetails} type="button">查看完整详情</button>
        ) : null}
      </header>
    ),
    filters: (
      <>
      <div className="activity-search-row">
        <label className="activity-search"><NovaIcon name="search" size={16} />
          <input aria-label="搜索日志" type="search" placeholder="搜索事件、来源或排查编号" value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <span role="status">显示 {filteredItems.length} / {items.length} 条</span>
      </div>
      <div className="activity-toolbar">
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
        <button
          className="console-button secondary activity-clear"
          disabled={clearing || items.length === 0}
          onClick={() => void clearHistory()}
          type="button"
        >
          <NovaIcon name="delete" size={15} />
          {clearing ? "正在清理…" : "清理历史"}
        </button>
      </div>
      {clearError ? <div className="activity-clear-error" role="alert"><strong>历史记录未清理</strong><span>{clearError}</span><small>记录已保留，可以重试。</small></div> : null}
      </>
    ),
    feed: (
      <section className="activity-list" aria-label="实时日志列表">
        <div className="activity-list-header" aria-hidden="true"><span /><span>等级</span><span>事件</span><span>来源</span><span>时间</span></div>
        <ul className="activity-feed" aria-label="最近活动">
          {filteredItems.length > 0 ? filteredItems.map((item) => (
            <li className="activity-feed-item" data-tone={item.tone ?? "error"} key={item.key}>
              <span className="activity-feed-marker" aria-hidden="true" />
              <span className="activity-level-tag" data-tone={item.tone ?? "error"}>{item.tone === "success" ? "完成" : item.tone === "info" ? "信息" : item.tone === "warn" ? "警告" : "故障"}</span>
              <div className="activity-item-copy">
                <strong>{item.title}</strong>
                <p>{item.detail}</p>
                {(item.count ?? 1) > 1 ? <small>本次会话重复 {item.count} 次</small> : null}
                {item.technicalDetail || item.requestId ? (
                  <details>
                    <summary>原始错误与开发者详情</summary>
                    {item.requestId ? <p>排查编号：{item.requestId}</p> : null}
                    {item.technicalDetail ? <pre>{item.technicalDetail}</pre> : null}
                  </details>
                ) : null}
              </div>
              <span className="activity-domain-tag">{item.tag || "NovaSight"}</span>
              {item.time ? <time dateTime={new Date(item.time).toISOString()} title={new Date(item.time).toLocaleString("zh-CN")}>{formatActivityTime(item.time)}</time> : <time>时间未提供</time>}
            </li>
          )) : (
            <li className="activity-empty">
              <EmptyState icon="logs" title={items.length > 0 ? "没有符合条件的日志" : "暂无运行记录"}
                description={items.length > 0 ? "试试其他关键词或日志等级。" : "运行、模型、采集或配置发生变化时，会按时间出现在这里。"}>
              {items.length > 0 ? <button className="console-button" onClick={() => { setQuery(""); setFilter("all"); }} type="button">重置筛选</button> : null}
              </EmptyState>
            </li>
          )}
        </ul>
      </section>
    ),
  };

  return (
    <section className="console-page activity-view" aria-labelledby="activity-view-title">
      {moduleOrder.map((moduleId) => <div data-module={moduleId} key={moduleId}>{modules[moduleId] ?? null}</div>)}
    </section>
  );
}
