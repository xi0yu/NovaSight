import { useMemo } from "react";

import { type PluginInfo } from "../../api";
import { EmptyState, InlineError, Panel, StatusIndicator } from "../../components/ui";

export function PluginsView({
  plugins,
  error
}: {
  plugins: PluginInfo[];
  error: string | undefined;
}) {
  const groups = useMemo(
    () => ({
      vision: plugins.filter((plugin) => plugin.kind === "vision"),
      control: plugins.filter((plugin) => plugin.kind === "control"),
      other: plugins.filter((plugin) => plugin.kind !== "vision" && plugin.kind !== "control")
    }),
    [plugins]
  );

  if (plugins.length === 0) {
    return (
      <Panel title="插件" eyebrow="算法链">
        <InlineError message={error} />
        <EmptyState title="没有加载插件" detail="后端返回的插件链为空。" />
      </Panel>
    );
  }

  return (
    <div className="view-grid plugins-grid">
      {error ? (
        <Panel title="插件请求" eyebrow="数据质量">
          <InlineError message={error} />
        </Panel>
      ) : null}
      <PluginGroup title="视觉分析" plugins={groups.vision} />
      <PluginGroup title="控制算法" plugins={groups.control} />
      {groups.other.length > 0 ? <PluginGroup title="其他" plugins={groups.other} /> : null}
    </div>
  );
}

function PluginGroup({ title, plugins }: { title: string; plugins: PluginInfo[] }) {
  return (
    <Panel title={`${title}插件`} eyebrow="已加载模块">
      {plugins.length > 0 ? (
        <div className="plugin-list">
          {plugins.map((plugin) => (
            <article className="plugin-row" key={plugin.plugin_id}>
              <div>
                <strong className="mono">{plugin.plugin_id}</strong>
                <span>{plugin.kind}</span>
              </div>
              <StatusIndicator tone={plugin.enabled ? "good" : "idle"}>
                {plugin.enabled ? "启用" : "停用"}
              </StatusIndicator>
            </article>
          ))}
        </div>
      ) : (
        <EmptyState title={`没有${title}插件`} detail="这个链路暂时没有模块。" />
      )}
    </Panel>
  );
}
