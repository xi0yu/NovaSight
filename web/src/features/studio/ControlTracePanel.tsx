import { NovaIcon, type NovaIconName } from "../../components/visual";
import type {
  ControlTraceState,
  ControlTraceStep,
  ControlTraceStepId,
  ControlTraceSummary
} from "./controlTrace";

const STEP_ICONS: Record<ControlTraceStepId, NovaIconName> = {
  batch: "batch",
  target: "target-lock",
  aim: "target-center",
  controller: "response-curve",
  gate: "command-queue",
  device: "hid"
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

function StepStateBadge({ state }: { state: ControlTraceState }) {
  return (
    <span className={`control-trace-state ${state}`}>
      <NovaIcon name={STATE_ICONS[state]} size={12} />
      {STATE_LABELS[state]}
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
          <StepStateBadge state={step.state} />
        </div>
        <b>{step.value}</b>
        <p>{step.detail}</p>
        <small>{step.evidence}</small>
      </div>
    </li>
  );
}

export function ControlTracePanel({ trace }: { trace: ControlTraceSummary }) {
  return (
    <section className={`control-trace-panel ${trace.state}`} aria-labelledby="control-trace-title">
      <header className="control-trace-header">
        <div>
          <span className="class-config-eyebrow">CONTROL TRACE</span>
          <h2 id="control-trace-title">{trace.title}</h2>
          <p>{trace.detail}</p>
        </div>
        <div className="control-trace-count" aria-label={`控制链路通过 ${trace.completed}/${trace.total}`}>
          <strong>{trace.completed}</strong>
          <span>/ {trace.total}</span>
          <small>阶段通过</small>
        </div>
      </header>

      <ol className="control-trace-steps" aria-label="控制链路阶段">
        {trace.steps.map((step) => (
          <ControlTraceStepCard key={step.id} step={step} />
        ))}
      </ol>

      <div className="control-trace-facts" aria-label="控制链路摘要">
        {trace.facts.map((fact) => (
          <div key={fact.label}>
            <span>{fact.label}</span>
            <strong>{fact.value}</strong>
            <small>{fact.detail}</small>
          </div>
        ))}
      </div>
    </section>
  );
}
