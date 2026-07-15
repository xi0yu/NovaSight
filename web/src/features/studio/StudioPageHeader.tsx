import { StatusBadge } from "../../components/visual";
import { CONSOLE_PAGE_METADATA, type ConsolePage } from "./StudioNavigation";

export function StudioPageHeader({
  page,
  runtimeRunning,
  realtimeConnected
}: {
  page: ConsolePage;
  runtimeRunning: boolean;
  realtimeConnected: boolean;
}) {
  const metadata = CONSOLE_PAGE_METADATA[page];
  return (
    <header className="console-page-header">
      <div className="console-page-heading">
        <span>{metadata.group}</span>
        <h1>{metadata.title}</h1>
        <p>{metadata.description}</p>
      </div>
      <div className="console-page-status" aria-label="当前运行状态">
        <StatusBadge
          status={realtimeConnected ? "normal" : "warning"}
          icon={realtimeConnected ? "data-stream" : "clock-alert"}
          label={realtimeConnected ? "实时数据" : "等待实时数据"}
          size="sm"
        />
        <StatusBadge
          status={runtimeRunning ? "running" : "waiting"}
          icon={runtimeRunning ? "activity-pulse" : "empty-circle"}
          label={runtimeRunning ? "主链运行中" : "主链未启动"}
          size="sm"
        />
      </div>
    </header>
  );
}
