import { NovaIcon } from "../../components/visual";
import { ModelSelectionPanel, type ModelSelectionPanelProps } from "./ModelSelectionPanel";

export function ModelWorkspace({
  activeModelName,
  panelProps,
}: {
  activeModelName: string;
  panelProps: ModelSelectionPanelProps;
}) {
  const activeFile = panelProps.activeArtifactPath.split(/[\\/]/).pop();
  return (
    <section className="model-workspace" aria-labelledby="model-workspace-title">
      <header className="model-workspace-header">
        <span className="model-manager-dialog-icon" aria-hidden="true">
          <NovaIcon name="models" size={22} strokeWidth={1.7} />
        </span>
        <div>
          <span className="class-config-eyebrow">模型管理</span>
          <h2 id="model-workspace-title">模型库</h2>
          <p>查找设备上的模型文件，整理推荐与标签；选择文件本身不会改变推理。</p>
        </div>
        <div className={`model-manager-active-pill ${activeFile ? "" : "inactive"}`}>
          <i aria-hidden="true" />
          <span>{activeFile ? panelProps.activeLoaded ? "运行已装载" : "已部署 · 未装载" : "未部署"}</span>
          <b title={panelProps.activeArtifactPath}>{activeFile || "尚无模型"}</b>
          {activeFile ? <small title={activeModelName}>项目：{activeModelName}</small> : null}
        </div>
      </header>
      <div className="model-workspace-body">
        <ModelSelectionPanel {...panelProps} />
      </div>
    </section>
  );
}
