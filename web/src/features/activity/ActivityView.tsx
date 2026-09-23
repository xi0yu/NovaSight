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
  time?: number;
  count?: number;
  tone?: "info" | "warn" | "error" | "success";
};

export function ActivityView({
  items,
  onOpenDetails,
}: {
  items: ActivityItem[];
  onOpenDetails: () => void;
}) {
  const attentionCount = items.filter((item) => item.tone === "warn" || item.tone === "error" || item.tone === undefined).length;
  return (
    <section className="console-page activity-view" aria-labelledby="activity-view-title">
      <header className={attentionCount > 0 ? "activity-summary needs-attention" : "activity-summary is-clear"}>
        <span className="activity-summary-icon" aria-hidden="true">
          <NovaIcon name={attentionCount > 0 ? "triangle-alert" : "shield-check"} size={22} />
        </span>
        <div>
          <span>现在</span>
          <h2 id="activity-view-title">{attentionCount > 0 ? `${attentionCount} 件事需要注意` : items.length > 0 ? "最近操作已完成" : "一切正常"}</h2>
          <p>{attentionCount > 0 ? "问题会保留在这里，直到恢复或由你清除本次会话的历史记录。" : "当前会话没有需要处理的故障。"}</p>
        </div>
        {attentionCount > 0 ? (
          <button className="console-button secondary" onClick={onOpenDetails} type="button">查看完整详情</button>
        ) : null}
      </header>
      <div className="activity-feed" aria-label="最近活动">
        {items.length > 0 ? items.map((item) => (
          <article className="activity-feed-item" data-tone={item.tone ?? "error"} key={item.key}>
            <span className="activity-feed-marker" aria-hidden="true" />
            <div>
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
            <span>最近没有需要处理的活动</span>
            <small>运行、模型或配置出现故障时，会按时间出现在这里。</small>
          </div>
        )}
      </div>
    </section>
  );
}
