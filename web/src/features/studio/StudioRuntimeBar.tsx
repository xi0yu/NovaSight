import { NovaIcon } from "../../components/visual";
import type { ConsolePage } from "./StudioNavigation";

type StudioRuntimeBarProps = {
  page: ConsolePage;
  runtimeAvailable: boolean;
  runtimeLifecycleActive: boolean;
  runtimeControlRequested: boolean;
  runtimeStopping: boolean;
  launchPending: boolean;
  diagnosticModeReady: boolean;
  busy: boolean;
  emergencyStopping: boolean;
  runtimeControlUnavailable: boolean;
  captureStatus: string;
  inferenceStatus: string;
  onToggle: () => void;
  onEmergencyStop: () => void;
};

export function StudioRuntimeBar({
  page,
  runtimeAvailable,
  runtimeLifecycleActive,
  runtimeControlRequested,
  runtimeStopping,
  launchPending,
  diagnosticModeReady,
  busy,
  emergencyStopping,
  runtimeControlUnavailable,
  captureStatus,
  inferenceStatus,
  onToggle,
  onEmergencyStop
}: StudioRuntimeBarProps) {
  if (page === "overview" || page === "params") return null;

  if (page === "control-test") {
    const title = !runtimeAvailable
      ? "运行状态不可用，硬件单步测试已锁定"
      : diagnosticModeReady
        ? "安全测试模式"
        : "主链未完全停止，硬件单步测试已锁定";
    const detail = !runtimeAvailable
      ? "重新取得 novasightd 状态后，系统才会判断是否允许发送测试命令。"
      : diagnosticModeReady
        ? "主链已停止；每次操作只发送一条受控移动命令。"
        : "停止主链并等待状态确认，避免自动控制与人工测试同时输出。";

    return (
      <section
        className={`studio-runtime-bar diagnostic ${diagnosticModeReady ? "ready" : "locked"}`}
        role="status"
      >
        <span className="studio-runtime-icon" aria-hidden="true">
          <NovaIcon name={diagnosticModeReady ? "shield-check" : "triangle-alert"} size={17} />
        </span>
        <div className="studio-runtime-copy">
          <strong>{title}</strong>
          <small>{detail}</small>
        </div>
        {runtimeLifecycleActive ? (
          <button
            className="console-button danger"
            disabled={busy || emergencyStopping || runtimeStopping || runtimeControlUnavailable}
            onClick={onToggle}
            type="button"
          >
            <NovaIcon name="stop" size={15} />
            {runtimeStopping ? "正在停止" : "停止主链"}
          </button>
        ) : null}
      </section>
    );
  }

  return (
    <section className="studio-runtime-bar" aria-label="主链运行控制">
      <span className="studio-runtime-icon" aria-hidden="true">
        <NovaIcon name="activity-pulse" size={17} />
      </span>
      <div className="studio-runtime-copy">
        <strong>{runtimeLifecycleActive ? "实时主链" : "主链待机"}</strong>
        <small>{`采集 ${captureStatus} · 推理 ${inferenceStatus}`}</small>
      </div>
      <button
        className={!runtimeAvailable ? "console-button" : runtimeLifecycleActive ? "console-button danger" : "console-button primary"}
        disabled={busy || emergencyStopping || runtimeStopping || runtimeControlUnavailable}
        onClick={onToggle}
        type="button"
      >
        <NovaIcon name={runtimeLifecycleActive ? "stop" : "start"} size={15} />
        {runtimeStopping
          ? "正在停止"
          : launchPending
            ? "正在启动"
            : !runtimeAvailable
              ? "等待状态"
              : runtimeControlRequested
                ? "停止运行"
                : "运行"}
      </button>
      {runtimeLifecycleActive ? (
        <button
          className="console-button danger studio-emergency-stop"
          disabled={emergencyStopping}
          onClick={onEmergencyStop}
          type="button"
        >
          <NovaIcon name="pause-output" size={15} />
          {emergencyStopping ? "紧急停止中" : "紧急停止"}
        </button>
      ) : null}
    </section>
  );
}
