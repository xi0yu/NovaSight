import { NovaIcon } from "../../components/visual";
import { ModelSelectionPanel, type ModelSelectionPanelProps } from "./ModelSelectionPanel";

import "./model-workspace.css";

export function ModelWorkspace({
  activeModelName,
  onOpenInference,
  panelProps,
}: {
  activeModelName: string;
  onOpenInference: () => void;
  panelProps: ModelSelectionPanelProps;
}) {
  const activeFile = panelProps.activeArtifactPath.split(/[\\/]/).pop();
  const deploymentState = activeFile
    ? panelProps.activeLoaded ? "loaded" : "deployed"
    : "empty";
  const hasDiscoveredModel = panelProps.modelCount > 0 || Boolean(activeFile);
  return (
    <section className="model-workspace" aria-labelledby="model-workspace-title">
      <header className="model-workspace-header">
        <div className="model-workspace-title">
          <span className="model-manager-dialog-icon" aria-hidden="true">
            <NovaIcon name="models" size={22} strokeWidth={1.7} />
          </span>
          <div>
            <h2 id="model-workspace-title">当前模型与设备文件</h2>
            <p>选择模型文件，系统会先检查是否可用，再允许部署到当前设备。</p>
          </div>
        </div>
        <div className={`model-production-slot ${deploymentState}`}>
          <div className="model-production-slot-heading">
            <span><i aria-hidden="true" /> 当前生产槽</span>
            <strong>{activeFile ? panelProps.activeLoaded ? "运行已装载" : "已部署，待装载" : "空槽"}</strong>
          </div>
          <b title={panelProps.activeArtifactPath}>{activeFile || "尚未部署模型"}</b>
          <div className="model-production-slot-facts">
            <span title={activeModelName}>项目 <strong>{activeModelName}</strong></span>
            <span>输入 <strong>{panelProps.runtimeInputShape || "未上报"}</strong></span>
            <span>后端 <strong>{panelProps.runtimeBackend || "未上报"}</strong></span>
          </div>
          <button className="console-button secondary" onClick={onOpenInference} type="button">
            查看实时推理
          </button>
        </div>
        <ol className="model-promotion-overview" aria-label="模型进入生产槽的步骤">
          <li className={hasDiscoveredModel ? "complete" : "current"}><span>1</span><b>发现文件</b><small>设备资产</small></li>
          <li className={activeFile ? "complete" : "pending"}><span>2</span><b>资格检查</b><small>输入与解析</small></li>
          <li className={activeFile ? "complete" : "pending"}><span>3</span><b>部署模型</b><small>写入运行配置</small></li>
          <li className={panelProps.activeLoaded ? "complete" : activeFile ? "current" : "pending"}><span>4</span><b>运行装载</b><small>真实状态读回</small></li>
        </ol>
      </header>
      <div className="model-workspace-body">
        <ModelSelectionPanel {...panelProps} />
      </div>
    </section>
  );
}
