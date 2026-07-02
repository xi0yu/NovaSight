import type { ReactNode } from "react";

type MetricCardTone = "good" | "warn" | "bad" | "idle";

type MetricCardProps = {
  label: ReactNode;
  value: ReactNode;
  detail?: ReactNode;
  tone?: MetricCardTone;
};

export function MetricCard({ label, value, detail, tone = "idle" }: MetricCardProps) {
  return (
    <div className={`metric-card tone-${tone}`}>
      <span className="metric-card-label">{label}</span>
      <strong className="metric-card-value">{value}</strong>
      {detail ? <span className="metric-card-detail">{detail}</span> : null}
    </div>
  );
}
