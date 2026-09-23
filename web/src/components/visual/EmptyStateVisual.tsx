import type { ReactNode } from "react";

import { NovaIcon, type NovaIconName } from "./NovaIcon";

type EmptyStateVisualProps = {
  icon?: NovaIconName;
  title: string;
  detail: string;
  command?: string;
  action?: ReactNode;
  secondaryAction?: ReactNode;
  size?: "sm" | "md" | "lg";
  className?: string;
};

export function EmptyStateVisual({
  icon = "empty-circle",
  title,
  detail,
  command,
  action,
  secondaryAction,
  size = "md",
  className,
}: EmptyStateVisualProps) {
  const resolvedClassName = ["ns-empty", `ns-empty-${size}`, className].filter(Boolean).join(" ");

  return (
    <div className={resolvedClassName}>
      <div className="ns-empty-visual" aria-hidden="true">
        <svg viewBox="0 0 160 120" fill="none" className="ns-empty-illustration">
          <rect x="31" y="24" width="98" height="68" rx="14" />
          <path d="M46 76c16-30 29-30 42-7s25 22 34-5" />
          <path d="M45 44h24M45 56h14" />
          <circle cx="111" cy="44" r="7" />
          <path d="M31 92l-14 12M129 92l14 12" />
        </svg>
        <span className="ns-empty-icon">
          <NovaIcon name={icon} size={28} strokeWidth={1.9} />
        </span>
      </div>
      <div className="ns-empty-copy">
        <strong>{title}</strong>
        <span>{detail}</span>
        {command ? <code>{command}</code> : null}
      </div>
      {action || secondaryAction ? (
        <div className="ns-empty-actions">
          {action}
          {secondaryAction}
        </div>
      ) : null}
    </div>
  );
}
