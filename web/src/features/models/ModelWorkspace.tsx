import { NovaIcon } from "../../components/visual";
import { ModelSelectionPanel, type ModelSelectionPanelProps } from "./ModelSelectionPanel";

export function ModelWorkspace({
  activeModelName,
  panelProps,
}: {
  activeModelName: string;
  panelProps: ModelSelectionPanelProps;
}) {
  return (
    <section className="model-workspace" aria-labelledby="model-workspace-title">
      <header className="model-workspace-header">
        <span className="model-manager-dialog-icon" aria-hidden="true">
          <NovaIcon name="models" size={22} strokeWidth={1.7} />
        </span>
        <div>
          <span className="class-config-eyebrow">模型管理</span>
          <h2 id="model-workspace-title">选择、验证并切换本机 Engine</h2>
          <p>切换由后端事务完成；页面刷新后以 active artifact 和运行快照为准。</p>
        </div>
        <div className="model-manager-active-pill">
          <i aria-hidden="true" />
          <span>当前</span>
          <b>{activeModelName}</b>
        </div>
      </header>
      <div className="model-workspace-body">
        <ModelSelectionPanel {...panelProps} />
      </div>
    </section>
  );
}
