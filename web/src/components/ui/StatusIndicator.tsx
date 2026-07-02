import type { ReactNode } from "react";

type StatusIndicatorProps = {
  tone: "good" | "warn" | "bad" | "idle";
  children: ReactNode;
};

export function StatusIndicator({ tone, children }: StatusIndicatorProps) {
  return (
    <span className={`status-pill ${tone}`}>
      <span className="status-dot" />
      {children}
    </span>
  );
}
