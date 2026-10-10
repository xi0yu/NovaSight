import type { ReactNode } from "react";

import { StatusBadge } from "../visual";
import { toneToStatus, type StatusTone } from "../../design/statusTokens";

type StatusIndicatorProps = {
  tone: StatusTone;
  children: ReactNode;
};

export function StatusIndicator({ tone, children }: StatusIndicatorProps) {
  return <StatusBadge className={`status-pill ${tone}`} status={toneToStatus(tone)} label={children} />;
}
