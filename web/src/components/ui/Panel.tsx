import type { ReactNode } from "react";

type PanelProps = {
  title: string;
  eyebrow?: string;
  action?: ReactNode;
  children: ReactNode;
};

export function Panel({ title, eyebrow, action, children }: PanelProps) {
  return (
    <section className="ui-panel">
      <div className="ui-panel-header">
        <div>
          {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
          <h2>{title}</h2>
        </div>
        {action ? <div className="ui-panel-action">{action}</div> : null}
      </div>
      <div className="ui-panel-body">{children}</div>
    </section>
  );
}
