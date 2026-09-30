import type { Ref } from "react";

export type ModelSwitchDialogStatus = "running" | "success" | "failed";

const MODEL_SWITCH_STAGES = [
  ["确认所选文件", "已选择 Engine 文件；此时尚未验证模型内容。"],
  ["登记模型引用", "取得设备返回的模型登记编号。"],
  ["等待设备切换回执", "设备完成验证、配置与切换后，统一返回结果。"]
] as const;

export function ModelSwitchDialog({
  completedStages,
  currentStage,
  detail,
  error,
  modelName,
  onClose,
  open,
  status,
  dialogRef
}: {
  completedStages: number;
  currentStage: number;
  detail: string;
  error: string;
  modelName: string;
  onClose: () => void;
  open: boolean;
  status: ModelSwitchDialogStatus;
  dialogRef: Ref<HTMLElement>;
}) {
  if (!open) return null;
  return (
    <div className="model-switch-dialog-layer">
      <section
        aria-labelledby="model-switch-dialog-title"
        aria-modal="true"
        className={`model-switch-dialog ${status}`}
        ref={dialogRef}
        role="dialog"
        tabIndex={-1}
      >
        <header className="model-switch-dialog-header">
          <div>
            <span>模型切换</span>
            <h2 id="model-switch-dialog-title">{status === "success" ? "模型切换已确认" : status === "failed" ? "模型切换未确认" : "验证并切换模型"}</h2>
            <p title={modelName}>{modelName}</p>
          </div>
          <strong>{status === "success" ? "已完成" : status === "failed" ? "待核对" : "处理中"}</strong>
        </header>
        <ol className="model-switch-stage-list">
          {MODEL_SWITCH_STAGES.map(([title, caption], index) => {
            const state = index < completedStages
              ? "success"
              : status === "failed" && index === currentStage
                ? "failed"
                : status === "running" && index === currentStage
                  ? "running"
                  : "pending";
            return (
              <li className={state} key={title}>
                <span className="model-switch-stage-marker" aria-hidden="true">
                  {state === "success" ? "✓" : state === "failed" ? "×" : ""}
                </span>
                <div>
                  <strong>{title}</strong>
                  <p>{caption}</p>
                </div>
              </li>
            );
          })}
        </ol>
        <div className={error ? "model-switch-dialog-detail error" : "model-switch-dialog-detail"} role={error ? "alert" : "status"}>
          {error || detail}
        </div>
        <footer>
          <small>{status === "running" ? "切换期间请保持页面打开，完成后可以关闭。" : "运行状态以设备最新回执为准。"}</small>
          <button
            className="console-button primary"
            disabled={status === "running"}
            onClick={onClose}
            type="button"
          >
            {status === "success" ? "完成" : "关闭"}
          </button>
        </footer>
      </section>
    </div>
  );
}
