import { StatusIndicator } from "../../components/ui";
import { NovaIcon } from "../../components/visual";

export function CurrentModelSummary({
  active,
  inputShape,
  modelName,
  outputShape,
  precision,
  onOpenManager
}: {
  active: boolean;
  inputShape: string;
  modelName: string;
  outputShape: string;
  precision: string;
  onOpenManager: () => void;
}) {
  return (
    <div className="current-model-summary">
      <div className="current-model-identity">
        <span className="current-model-engine-mark" aria-hidden="true">
          <NovaIcon name="engine" size={24} strokeWidth={1.7} />
        </span>
        <div>
          <span className="class-config-eyebrow">ACTIVE INFERENCE MODEL</span>
          <h3 title={modelName}>{modelName}</h3>
        </div>
        <StatusIndicator tone={active ? "good" : "idle"}>
          {active ? "当前使用" : "未加载"}
        </StatusIndicator>
      </div>
      <dl className="current-model-runtime-facts">
        <div><dt>模型输入</dt><dd>{inputShape || "-"}</dd></div>
        <div><dt>输出形状</dt><dd>{outputShape || "-"}</dd></div>
        <div><dt>精度声明</dt><dd>{precision || "-"}</dd></div>
      </dl>
      <button className="console-button primary current-model-manage-button" onClick={onOpenManager} type="button">
        <NovaIcon name="model-switch" size={16} />
        管理与切换模型
      </button>
    </div>
  );
}
