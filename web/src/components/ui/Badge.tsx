import type { ReactNode } from "react";

type BadgeProps = {
  tone?: "default" | "good" | "warn" | "bad" | "idle";
  className?: string;
  children: ReactNode;
};

export function Badge({ tone = "default", className, children }: BadgeProps) {
  const resolvedClassName = ["badge", tone !== "default" ? `badge-${tone}` : "", className]
    .filter(Boolean)
    .join(" ");

  return <span className={resolvedClassName}>{children}</span>;
}
