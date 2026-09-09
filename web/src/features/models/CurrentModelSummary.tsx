import { StatusIndicator } from "../../components/ui";
import { NovaIcon } from "../../components/visual";

export function CurrentModelSummary({
  artifactKind,
  deployed,
  inputShape,
  loaded,
  modelName,
  onOpenManager
}: {
  artifactKind: string;
  deployed: boolean;
  inputShape: string;
  loaded: boolean;
  modelName: string;
  onOpenManager: () => void;
}) {
  return (
    <div className="current-model-summary">
      <div className="current-model-identity">
        <span className="current-model-engine-mark" aria-hidden="true">
          <NovaIcon name="engine" size={24} strokeWidth={1.7} />
        </span>
        <div>
          <span className="class-config-eyebrow">当前运行模型</span>
          <h3 title={modelName}>{modelName}</h3>
        </div>
        <StatusIndicator tone={loaded ? "good" : "idle"}>
          {loaded ? "运行已装载" : deployed ? "已部署 · 未装载" : "未部署"}
        </StatusIndicator>
      </div>
      <dl className="current-model-runtime-facts">
        <div><dt>登记输入</dt><dd>{inputShape || "-"}</dd></div>
        <div><dt>运行状态</dt><dd>{loaded ? "已装载" : "未装载"}</dd></div>
        <div><dt>模型文件</dt><dd>{artifactKind || "-"}</dd></div>
      </dl>
      <button className="console-button primary current-model-manage-button" onClick={onOpenManager} type="button">
        <NovaIcon name="model-switch" size={16} />
        管理与切换模型
      </button>
    </div>
  );
}
