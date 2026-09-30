import { NovaIcon } from "../../components/visual";
import type { ConsolePage } from "./StudioNavigation";

type StudioRuntimeBarProps = {
  page: ConsolePage;
  runtimeAvailable: boolean;
  runtimeLifecycleActive: boolean;
  diagnosticModeReady: boolean;
  captureStatus: string;
  inferenceStatus: string;
};

export function StudioRuntimeBar({
  page,
  runtimeAvailable,
  runtimeLifecycleActive,
  diagnosticModeReady,
  captureStatus,
  inferenceStatus
}: StudioRuntimeBarProps) {
  if (!(["capture", "infer", "control", "latency", "control-test"] as ConsolePage[]).includes(page)) return null;
  if (page === "capture" && runtimeAvailable) return null;

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
        {runtimeLifecycleActive ? <span className="studio-runtime-state">请先到首页关闭运行</span> : null}
      </section>
    );
  }

  return (
    <section className="studio-runtime-bar" aria-label="主链运行状态">
      <span className="studio-runtime-icon" aria-hidden="true">
        <NovaIcon name="activity-pulse" size={17} />
      </span>
      <div className="studio-runtime-copy">
        <strong>{!runtimeAvailable ? "运行状态未确认" : runtimeLifecycleActive ? "实时主链" : "主链待机"}</strong>
        <small>{`采集 ${captureStatus} · 推理 ${inferenceStatus}`}</small>
      </div>
      <span className="studio-runtime-state">{runtimeLifecycleActive ? "运行由首页管理" : "待机"}</span>
    </section>
  );
}
