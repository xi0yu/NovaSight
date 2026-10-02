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
