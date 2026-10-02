import type { ReactNode } from "react";
import { NovaIcon, type NovaIconName } from "../visual";
import "./empty-state.css";
import chibiArt from "../../assets/themes/nova-chibi.webp";

// Adapted from shadcn/ui Empty (MIT). Kept as one component with native CSS;
// source and license are included in public/third-party-ui.txt.
export function EmptyState({ icon, title, description, children }: {
  icon: NovaIconName;
  title: string;
  description: string;
  children?: ReactNode;
}) {
  return <div className="ui-empty" data-slot="empty">
    <div data-slot="empty-header">
      <div className="ui-empty-illustration" aria-hidden="true"><img src={chibiArt} width="100" height="112" alt="" loading="lazy" /><span data-slot="empty-icon"><NovaIcon name={icon} size={20} /></span></div>
      <strong data-slot="empty-title">{title}</strong>
      <p data-slot="empty-description">{description}</p>
    </div>
    {children ? <div data-slot="empty-content">{children}</div> : null}
  </div>;
}
