import type { ReactNode } from "react";

import { EmptyStateVisual, type NovaIconName } from "../visual";

type EmptyStateProps = {
  title: string;
  detail: string;
  command?: string;
  action?: ReactNode;
  secondaryAction?: ReactNode;
  icon?: NovaIconName;
};

export function EmptyState({
  title,
  detail,
  command,
  action,
  secondaryAction,
  icon,
}: EmptyStateProps) {
  return (
    <EmptyStateVisual
      action={action}
      command={command}
      detail={detail}
      icon={icon}
      secondaryAction={secondaryAction}
      title={title}
    />
  );
}
