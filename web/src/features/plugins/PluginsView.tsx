import { useMemo } from "react";

import { type PluginInfo, type RuntimeState } from "../../api";
import { Badge, EmptyState, InlineError, Panel, StatusIndicator } from "../../components/ui";
import { Field } from "../shared/Field";

export function PluginsView({
  plugins,
  runtime,
  error
}: {
  plugins: PluginInfo[];
  runtime: RuntimeState | null;
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
      <div className="view-grid plugins-grid">
        <AlgorithmConfigPanel config={runtime?.config ?? null} />
        <Panel title="插件" eyebrow="算法链">
          <InlineError message={error} />
          <EmptyState title="没有加载插件" detail="后端返回的插件链为空。" />
        </Panel>
      </div>
    );
  }

  return (
    <div className="view-grid plugins-grid">
      <AlgorithmConfigPanel config={runtime?.config ?? null} />
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

function AlgorithmConfigPanel({ config }: { config: Record<string, unknown> | null }) {
  const control = isRecord(config?.control) ? config.control : {};
  const strategy = String(control.strategy ?? "pid");
  const outputMode = String(control.output_mode || "跟随执行器");
  const maxAbsDx = String(control.max_abs_dx ?? "未设置");
  const maxAbsDy = String(control.max_abs_dy ?? "未设置");
  const minConfidence = String(control.min_confidence ?? "未设置");
  const interval = String(control.command_interval_ms ?? "未设置");

  return (
    <Panel title="算法参数" eyebrow="运行配置映射">
      <div className="algorithm-config-grid">
        <article className={strategy === "pid" ? "algorithm-card active" : "algorithm-card"}>
          <div className="algorithm-card-head">
            <strong>PID 平滑追踪</strong>
            <Badge tone={strategy === "pid" ? "good" : "idle"}>
              {strategy === "pid" ? "当前策略" : "可选策略"}
            </Badge>
          </div>
          <div className="field-grid compact">
            <Field label="X 限幅" value={maxAbsDx} mono />
            <Field label="Y 限幅" value={maxAbsDy} mono />
            <Field label="最低置信度" value={minConfidence} mono />
            <Field label="合并间隔" value={`${interval} ms`} mono />
          </div>
          <p>这些参数来自运行配置页，保存后由后端线程安全快照读取。</p>
        </article>

        <article className={strategy === "predictive" ? "algorithm-card active" : "algorithm-card"}>
          <div className="algorithm-card-head">
            <strong>预测追踪</strong>
            <Badge tone={strategy === "predictive" ? "good" : "idle"}>
              {strategy === "predictive" ? "当前策略" : "可选策略"}
            </Badge>
          </div>
          <div className="field-grid compact">
            <Field label="输出模式" value={outputMode} mono />
            <Field label="触发来源" value="硬件盒子回传" />
            <Field label="限幅策略" value={`${maxAbsDx} / ${maxAbsDy}`} mono />
            <Field label="发送节流" value={`${interval} ms`} mono />
          </div>
          <p>插件页展示算法配置归属；参数编辑统一进入运行配置，避免两套配置不一致。</p>
        </article>
      </div>
    </Panel>
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
