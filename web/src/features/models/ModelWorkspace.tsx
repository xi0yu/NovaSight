import { ModelSelectionPanel, type ModelSelectionPanelProps } from "./ModelSelectionPanel";

export function ModelWorkspace({
  activeModelName,
  activeArtifactStatus,
  onOpenInference,
  panelProps,
}: {
  activeModelName: string;
  activeArtifactStatus: string | null;
  onOpenInference: () => void;
  panelProps: ModelSelectionPanelProps;
}) {
  const activeFile = panelProps.activeArtifactPath.split(/[\\/]/).pop();
  const deploymentState = activeFile
    ? panelProps.activeLoaded ? "loaded" : "deployed"
    : "empty";
  return (
    <section className="model-workspace" aria-labelledby="model-workspace-title">
      <header className="model-workspace-header">
        <div className={`model-production-slot ${deploymentState}`}>
          <div className="model-production-slot-heading">
            <h2 id="model-workspace-title"><i aria-hidden="true" /> 当前使用的模型</h2>
            <strong>{activeFile ? panelProps.activeLoaded ? "运行已装载" : "已部署，待装载" : "尚未选择"}</strong>
          </div>
          <b title={panelProps.activeArtifactPath}>{activeFile || "尚未部署模型"}</b>
          <div className="model-production-slot-facts">
            <span title={activeModelName}>项目 <strong>{activeModelName}</strong></span>
            <span>验证 <strong>{activeArtifactStatus === "ready" ? "已有验证记录" : "待验证"}</strong></span>
            <span>输入 <strong>{panelProps.runtimeInputShape || "未上报"}</strong></span>
            <span>后端 <strong>{panelProps.runtimeBackend || "未上报"}</strong></span>
          </div>
          <button className="console-button secondary" onClick={onOpenInference} type="button">
            {activeFile ? "查看实时推理" : "查看推理状态"}
          </button>
        </div>
      </header>
      <div className="model-workspace-body">
        <ModelSelectionPanel {...panelProps} />
      </div>
    </section>
  );
}
