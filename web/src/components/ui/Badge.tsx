import type { ReactNode } from "react";

import { NovaIcon, type NovaIconName } from "../visual";

type BadgeProps = {
  tone?: "default" | "good" | "warn" | "bad" | "idle";
  className?: string;
  icon?: NovaIconName;
  children: ReactNode;
};

export function Badge({ tone = "default", className, icon, children }: BadgeProps) {
  const resolvedClassName = ["badge", tone !== "default" ? `badge-${tone}` : "", className]
    .filter(Boolean)
    .join(" ");

  return (
    <span className={resolvedClassName}>
      {icon ? <NovaIcon name={icon} size={14} /> : null}
      {children}
    </span>
  );
}
