import type { Ref } from "react";

export type ModelSwitchDialogStatus = "running" | "success" | "failed";

const MODEL_SWITCH_STAGES = [
  ["确认 Engine 文件", "仅按 .engine 后缀接受候选，目录浏览阶段不会加载模型。"],
  ["登记模型引用", "复用已有登记；未登记时只按路径和文件元数据创建轻量引用。"],
  ["验证模型输入输出", "读取真实输入、输出、尺寸与数据类型，非法 Engine 在此终止。"],
  ["准备运行配置", "复用匹配配置；缺失或不匹配时自动生成运行配置。"],
  ["切换推理运行态", "应用模型，并在主链运行时等待新的识别结果。"]
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
  const progress = status === "success"
    ? 100
    : Math.round((completedStages / MODEL_SWITCH_STAGES.length) * 100);

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
            <span>MODEL ACTIVATION</span>
            <h2 id="model-switch-dialog-title">验证并切换模型</h2>
            <p title={modelName}>{modelName}</p>
          </div>
          <strong>{progress}%</strong>
        </header>
        <div className="model-switch-progress" aria-hidden="true">
          <i style={{ width: `${progress}%` }} />
        </div>
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
        <div className={error ? "model-switch-dialog-detail error" : "model-switch-dialog-detail"}>
          {error || detail}
        </div>
        <footer>
          <small>模型验证、运行配置准备和运行态切换由后端自动完成。</small>
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
