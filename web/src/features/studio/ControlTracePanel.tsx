import { NovaIcon, type NovaIconName } from "../../components/visual";
import type {
  ControlTraceState,
  ControlTraceStep,
  ControlTraceStepId,
  ControlTraceSummary
} from "./controlTrace";

const STEP_ICONS: Record<ControlTraceStepId, NovaIconName> = {
  target: "target-lock",
  trigger: "clock",
  aim: "target-center",
  controller: "response-curve",
  recoil: "activity-pulse",
  limiter: "control",
  output: "hid"
};

const STATE_ICONS: Record<ControlTraceState, NovaIconName> = {
  ready: "check-circle",
  blocked: "triangle-alert",
  waiting: "clock",
  idle: "empty-circle"
};

const STATE_LABELS: Record<ControlTraceState, string> = {
  ready: "通过",
  blocked: "阻断",
  waiting: "等待",
  idle: "空闲"
};

function StepStateBadge({ state, label }: { state: ControlTraceState; label?: string }) {
  return (
    <span className={`control-trace-state ${state}`}>
      <NovaIcon name={STATE_ICONS[state]} size={12} />
      {label ?? STATE_LABELS[state]}
    </span>
  );
}

function ControlTraceStepCard({ step }: { step: ControlTraceStep }) {
  return (
    <li className={`control-trace-step ${step.state}`}>
      <span className="control-trace-step-icon" aria-hidden="true">
        <NovaIcon name={STEP_ICONS[step.id]} size={18} />
      </span>
      <div className="control-trace-step-copy">
        <div className="control-trace-step-heading">
          <strong>{step.label}</strong>
          <StepStateBadge state={step.state} label={step.stateLabel} />
        </div>
        <b>{step.value}</b>
      </div>
    </li>
  );
}

export function ControlTracePanel({ trace }: { trace: ControlTraceSummary }) {
  return (
    <section className={`control-trace-panel ${trace.state}`} aria-labelledby="control-trace-title">
      <header className="control-trace-header">
        <div>
          <span className="class-config-eyebrow">实时控制链</span>
          <h2 id="control-trace-title">{trace.title}</h2>
        </div>
        <div className="control-trace-count" aria-label={`控制链路通过 ${trace.completed}/${trace.total}`}>
          <strong>{trace.completed}</strong>
          <span>/ {trace.total}</span>
        </div>
      </header>

      <ol className="control-trace-steps" aria-label="控制链路阶段">
        {trace.steps.map((step) => (
          <ControlTraceStepCard key={step.id} step={step} />
        ))}
      </ol>
    </section>
  );
}
