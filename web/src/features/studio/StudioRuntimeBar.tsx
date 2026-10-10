import { NovaIcon } from "../../components/visual";
import type { ConsolePage } from "./StudioNavigation";

type StudioRuntimeBarProps = {
  page: ConsolePage;
  runtimeAvailable: boolean;
  runtimeLifecycleActive: boolean;
  captureStatus: string;
  inferenceStatus: string;
};

export function StudioRuntimeBar({
  page,
  runtimeAvailable,
  runtimeLifecycleActive,
  captureStatus,
  inferenceStatus
}: StudioRuntimeBarProps) {
  if (!(["capture", "infer", "control", "latency"] as ConsolePage[]).includes(page)) return null;
  if (page === "capture" && runtimeAvailable) return null;


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
