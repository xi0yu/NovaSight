import { EmptyState, InlineError, Panel, StatusIndicator } from "../../components/ui";
import {
  type ExecutorStatus,
  type HealthResponse,
  type RuntimeState
} from "../../api";
import { Field } from "../shared/Field";
import { formatProfile, statusTone } from "../shared/format";
import { InferenceControl } from "../shared/InferenceControl";

type DashboardViewProps = {
  health: HealthResponse | null;
  runtime: RuntimeState | null;
  loading: boolean;
  errors: {
    health?: string;
    runtime?: string;
  };
  onRefresh: () => void;
  onInferenceControlCommand: (action: "start" | "stop") => void;
  runtimeCommandBusy: boolean;
};

function OverviewView({
  health,
  runtime,
  loading,
  errors,
  onRefresh,
  onInferenceControlCommand,
  runtimeCommandBusy
}: DashboardViewProps) {
  const executor = runtime?.executor;
  const selectedExecutor = executor?.selected;
  const selectedAvailability = selectedExecutor
    ? executor?.executors[selectedExecutor]?.available
    : undefined;

  return (
    <div className="view-grid dashboard-grid">
      <Panel
        title="运行总览"
        eyebrow="核心状态"
        action={
          <div className="panel-actions">
            <InferenceControl
              busy={runtimeCommandBusy}
              running={Boolean(runtime?.running)}
              onCommand={onInferenceControlCommand}
            />
            <button className="button" type="button" onClick={onRefresh}>
              刷新
            </button>
          </div>
        }
      >
        <InlineError message={errors.health ?? errors.runtime} />
        <div className="metric-grid">
          <div className="metric">
            <span>后端</span>
            <StatusIndicator tone={statusTone(health?.ok)}>
              {loading ? "检查中" : health?.ok ? "已连接" : "离线"}
            </StatusIndicator>
          </div>
          <div className="metric">
            <span>运行态</span>
            <StatusIndicator tone={runtime?.running ? "good" : "idle"}>
              {runtime?.running ? "运行中" : "待机"}
            </StatusIndicator>
          </div>
          <div className="metric">
            <span>采集配置</span>
            <strong>{formatProfile(runtime?.capture)}</strong>
          </div>
          <div className="metric">
            <span>控制输出</span>
            <StatusIndicator tone={statusTone(selectedAvailability)}>
              {selectedExecutor ?? "未选择"}
            </StatusIndicator>
          </div>
        </div>
        <div className="field-grid">
          <Field label="采集设备" value={runtime?.capture?.device ?? "/dev/video0"} mono />
          <Field label="采集后端" value={runtime?.capture?.backend ?? "未打开"} mono />
          <Field label="当前模型" value={runtime?.active_model?.project?.name ?? "未发布模型"} />
          <Field
            label="模型文件"
            value={runtime?.active_model?.artifact?.path ?? "未绑定"}
            mono
          />
        </div>
      </Panel>

      <Panel title="输出执行器" eyebrow="可用性">
        <InlineError message={errors.runtime} />
        <ExecutorTable executor={executor} />
      </Panel>
    </div>
  );
}

function ExecutorTable({ executor }: { executor: ExecutorStatus | undefined }) {
  const entries = Object.entries(executor?.executors ?? {});

  if (entries.length === 0) {
    return <EmptyState title="没有执行器状态" detail="后端没有返回控制输出执行器。" />;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>执行器</th>
            <th>当前选择</th>
            <th>可用</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([id, state]) => (
            <tr key={id}>
              <td className="mono">{id}</td>
              <td>{executor?.selected === id ? "是" : "否"}</td>
              <td>
                <StatusIndicator tone={state.available ? "good" : "bad"}>
                  {state.available ? "可用" : "不可用"}
                </StatusIndicator>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DashboardView(props: DashboardViewProps) {
  return <OverviewView {...props} />;
}
