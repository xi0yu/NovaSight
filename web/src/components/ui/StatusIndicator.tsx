import type { ReactNode } from "react";

import { StatusBadge } from "../visual";
import { toneToStatus, type LegacyTone } from "../../design/statusTokens";

type StatusIndicatorProps = {
  tone: LegacyTone;
  children: ReactNode;
};

export function StatusIndicator({ tone, children }: StatusIndicatorProps) {
  return <StatusBadge className={`status-pill ${tone}`} status={toneToStatus(tone)} label={children} />;
}
