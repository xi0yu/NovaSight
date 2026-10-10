import { useMemo, useRef, useState, type ReactNode } from "react";

import { NovaIcon } from "../../components/visual";
import { EmptyState } from "../../components/ui/EmptyState";
import { ListSkeleton } from "../../components/ui";
import { getErrorMessage } from "../shared/format";
import type { ActivityConnectionState } from "./useActivityStream";
import { reportSuccess } from "../../lib/toast";

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

function ActivityCopy({ item, onInspect }: { item: ActivityItem; onInspect: () => void }) {
  const [copying, setCopying] = useState(false);
  const [manualCopy, setManualCopy] = useState(false);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const text = [item.time ? new Date(item.time).toISOString() : "时间未提供", item.tag || "NovaSight",
    item.title, item.detail, item.count && item.count > 1 ? `重复 ${item.count} 次` : "",
    item.requestId ? `排查编号：${item.requestId}` : "", item.technicalDetail].filter(Boolean).join("\n");
  const copy = async () => {
    onInspect();
    setCopying(true);
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(text);
      setManualCopy(false);
      reportSuccess("日志已复制", item.title, "activity-copy", "feedback");
    } catch {
      setManualCopy(true);
      window.requestAnimationFrame(() => { textRef.current?.focus(); textRef.current?.select(); });
    } finally { setCopying(false); }
  };
  return <div className="activity-copy">
    <button className="console-button" type="button" aria-label={`复制日志：${item.title}`} aria-busy={copying} disabled={copying} onClick={() => void copy()}>
      <NovaIcon name="copy" size={13} />{copying ? "正在复制…" : "复制记录"}
    </button>
    {manualCopy ? <div className="activity-copy-manual" role="status">
      <span>浏览器未允许自动复制。内容已选中，按 Ctrl+C / ⌘C 复制。</span>
      <textarea ref={textRef} aria-label={`手动复制日志：${item.title}`} readOnly value={text} rows={4} />
    </div> : null}
  </div>;
}

export function ActivityView({
  items,
  onClear,
  onOpenDetails,
  moduleOrder = DEFAULT_MODULES,
  connectionState,
  connectionError,
  onRetry,
  loading = false,
}: {
  items: ActivityItem[];
  onClear: () => Promise<void> | void;
  onOpenDetails: () => void;
  moduleOrder?: readonly string[];
  connectionState?: ActivityConnectionState;
  connectionError?: string;
  onRetry?: () => void;
  loading?: boolean;
}) {
  const [filter, setFilter] = useState<ActivityFilter>("all");
  const [clearing, setClearing] = useState(false);
  const [query, setQuery] = useState("");
  const [clearError, setClearError] = useState("");
  const [source, setSource] = useState("all");
  const [pausedItems, setPausedItems] = useState<ActivityItem[] | null>(null);
  const listRef = useRef<HTMLElement>(null);
  const clearingRef = useRef(false);
  const shownItems = pausedItems ?? items;
  const sourceOptions = useMemo(() => [...new Set([...shownItems, ...items].map((item) => item.tag || "NovaSight"))].sort(), [shownItems, items]);
  const pausedKeys = new Set(pausedItems?.map((item) => item.key));
  const unseenCount = pausedItems ? items.filter((item) => !pausedKeys.has(item.key)).length : 0;
  const attentionCount = shownItems.filter((item) => item.tone === "warn" || item.tone === "error" || item.tone === undefined).length;
  const completedCount = shownItems.filter((item) => item.tone === "success").length;
  const filteredItems = useMemo(() => shownItems.filter((item) =>
    (filter === "all" || (item.tone ?? "error") === filter)
    && (source === "all" || (item.tag || "NovaSight") === source)
    && [item.title, item.detail, item.tag, item.requestId, item.technicalDetail].join(" ").toLocaleLowerCase().includes(query.trim().toLocaleLowerCase())
  ), [filter, shownItems, query, source]);
  const clearHistory = async () => {
    if (clearingRef.current) return;
    clearingRef.current = true;
    setClearing(true);
    setClearError("");
    try { await onClear(); setPausedItems(null); reportSuccess("历史记录已清理", "新事件仍会继续记录。", "activity-clear", "feedback"); }
    catch (error) { setClearError(getErrorMessage(error)); }
    finally { clearingRef.current = false; setClearing(false); }
  };
  const counts = useMemo(() => ({
    all: shownItems.length,
    error: shownItems.filter((item) => (item.tone ?? "error") === "error").length,
    warn: shownItems.filter((item) => item.tone === "warn").length,
    success: shownItems.filter((item) => item.tone === "success").length,
    info: shownItems.filter((item) => item.tone === "info").length,
  }), [shownItems]);

  const modules: Record<string, ReactNode> = {
    summary: (
      <header className={attentionCount > 0 ? "activity-summary needs-attention" : loading && shownItems.length === 0 ? "activity-summary is-pending" : "activity-summary is-clear"} aria-live="polite">
        <span className="activity-summary-icon" aria-hidden="true">
          <NovaIcon name={attentionCount > 0 ? "triangle-alert" : loading && shownItems.length === 0 ? "clock" : "shield-check"} size={22} />
        </span>
        <div>
          <span>活动记录 · {loading && shownItems.length === 0 ? "待读取" : `${shownItems.length} 条`}</span>
          <h2 id="activity-view-title">{attentionCount > 0 ? `${attentionCount} 条故障与警告记录` : completedCount > 0 ? "最近操作已完成" : shownItems.length > 0 ? "最近有新记录" : loading ? "正在读取日志" : connectionError ? "日志尚未读取" : "暂无活动记录"}</h2>
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
        <label className="activity-source">来源<select aria-label="日志来源" value={source} onChange={(event) => setSource(event.target.value)}><option value="all">全部来源</option>{sourceOptions.map((tag) => <option key={tag} value={tag}>{tag}</option>)}</select></label>
        <button className="console-button" type="button" aria-pressed={pausedItems !== null} onClick={() => {
          if (pausedItems) { listRef.current?.scrollTo({ top: 0 }); setPausedItems(null); }
          else setPausedItems([...items]);
        }}>{pausedItems ? `恢复实时${unseenCount > 0 ? ` · ${unseenCount} 条新记录` : ""}` : "暂停查看"}</button>
        <span role="status">{loading && shownItems.length === 0 ? "正在读取" : `${pausedItems ? "已暂停查看 · " : ""}显示 ${filteredItems.length} / ${shownItems.length} 条`}</span>
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
              <span>{label}</span><b>{loading && shownItems.length === 0 ? "—" : counts[value]}</b>
            </button>
          ))}
        </nav>
        <button
          className="console-button secondary activity-clear"
          disabled={clearing || loading || items.length === 0}
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
      <section className="activity-list" aria-label="实时日志列表" ref={listRef} onScroll={(event) => {
        if (pausedItems === null && event.currentTarget.scrollTop > 24) setPausedItems([...items]);
      }}>
        <div className="activity-list-header" aria-hidden="true"><span /><span>等级</span><span>事件</span><span>来源</span><span>时间</span></div>
        {loading && shownItems.length === 0 ? <ListSkeleton label="正在读取日志记录" /> : <ul className="activity-feed" aria-label="最近活动">
          {filteredItems.length > 0 ? filteredItems.map((item) => (
            <li className="activity-feed-item" data-tone={item.tone ?? "error"} key={item.key}>
              <span className="activity-feed-marker" aria-hidden="true" />
              <span className="activity-level-tag" data-tone={item.tone ?? "error"}>{item.tone === "success" ? "完成" : item.tone === "info" ? "信息" : item.tone === "warn" ? "警告" : "故障"}</span>
              <div className="activity-item-copy">
                <div className="activity-event-header"><strong>{item.title}</strong><ActivityCopy item={item} onInspect={() => { if (pausedItems === null) setPausedItems([...items]); }} /></div>
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
              <EmptyState icon="logs" title={shownItems.length > 0 ? "没有符合条件的日志" : connectionError ? "日志尚未读取" : "暂无运行记录"}
                description={shownItems.length > 0 ? "试试其他关键词、来源或日志等级。" : connectionError ? "连接恢复后重新读取，当前没有可展示的记录。" : "运行、模型、采集或配置发生变化时，会按时间出现在这里。"}>
              {shownItems.length > 0 ? <button className="console-button" onClick={() => { setQuery(""); setFilter("all"); setSource("all"); }} type="button">重置筛选</button> : null}
              </EmptyState>
            </li>
          )}
        </ul>}
      </section>
    ),
  };

  return (
    <section className="console-page activity-view" aria-label="实时日志">
      {connectionState ? <div className="activity-connection" data-state={connectionState} role="status">
        <NovaIcon name={connectionState === "connected" ? "check-circle" : "signal-lost"} size={16} />
        <span>{connectionState === "connected" ? "实时日志已连接" : connectionState === "loading" ? "正在读取日志并连接实时通道…" : connectionState === "paused" ? "等待重新授权，日志更新已暂停" : connectionError || "日志状态未确认，已有记录会保留。"}</span>
        {(connectionState === "error" || connectionState === "disconnected") && onRetry ? <button className="console-button" type="button" onClick={onRetry}>重新读取日志</button> : null}
      </div> : null}
      {moduleOrder.map((moduleId) => <div data-module={moduleId} key={moduleId}>{modules[moduleId] ?? null}</div>)}
    </section>
  );
}
