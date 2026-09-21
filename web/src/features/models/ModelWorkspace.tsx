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
          <h2 id="model-workspace-title">找到模型，确认后切换</h2>
          <p>从设备目录查找 .engine，核对右侧文件后再验证切换。</p>
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
