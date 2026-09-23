import type { ReactNode } from "react";

import { statusSpecs, type NovaStatus } from "../../design/statusTokens";
import { NovaIcon, type NovaIconName } from "./NovaIcon";

type StatusBadgeProps = {
  status: NovaStatus;
  label: ReactNode;
  detail?: ReactNode;
  icon?: NovaIconName;
  size?: "sm" | "md" | "lg";
  className?: string;
};

export function StatusBadge({
  status,
  label,
  detail,
  icon,
  size = "md",
  className,
}: StatusBadgeProps) {
  const spec = statusSpecs[status];
  const resolvedClassName = [
    "ns-status",
    `ns-status-${spec.className}`,
    `ns-status-${size}`,
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <span className={resolvedClassName}>
      <NovaIcon name={icon ?? spec.icon} size={size === "lg" ? 18 : size === "sm" ? 14 : 16} />
      <span className="ns-status-label">{label}</span>
      {detail ? <span className="ns-status-detail">{detail}</span> : null}
    </span>
  );
}

type StatusCardProps = {
  status: NovaStatus;
  title: string;
  description: ReactNode;
  metric?: ReactNode;
  action?: ReactNode;
  icon?: NovaIconName;
  className?: string;
};

export function StatusCard({
  status,
  title,
  description,
  metric,
  action,
  icon,
  className,
}: StatusCardProps) {
  const spec = statusSpecs[status];
  const resolvedClassName = ["ns-status-card", `ns-status-card-${spec.className}`, className]
    .filter(Boolean)
    .join(" ");

  return (
    <section className={resolvedClassName}>
      <div className="ns-status-card-icon">
        <NovaIcon name={icon ?? spec.icon} size={22} strokeWidth={1.9} />
      </div>
      <div className="ns-status-card-copy">
        <strong>{title}</strong>
        <span>{description}</span>
        {metric ? <code>{metric}</code> : null}
      </div>
      {action ? <div className="ns-status-card-action">{action}</div> : null}
    </section>
  );
}
