import type { ReactNode } from "react";

import { studioNavItems, type StudioViewId } from "./navigation";

type StudioShellProps = {
  activeView: StudioViewId;
  title: string;
  subtitle: string;
  status?: ReactNode;
  children: ReactNode;
  onNavigate: (view: StudioViewId) => void;
};

export function StudioShell({
  activeView,
  title,
  subtitle,
  status,
  children,
  onNavigate
}: StudioShellProps) {
  return (
    <div className="studio-shell">
      <aside className="studio-sidebar">
        <div className="studio-brand">
          <span className="brand-mark"><span>NS</span></span>
          <div className="studio-brand-copy">
            <strong>NovaSight Studio</strong>
            <span>Production console</span>
          </div>
        </div>

        <div className="studio-sidebar-section">
          <span className="studio-sidebar-label">Workspace</span>
          <nav className="studio-nav" aria-label="Studio navigation">
            {studioNavItems.map((item) => (
              <button
                key={item.id}
                type="button"
                className={item.id === activeView ? "nav-item active" : "nav-item"}
                aria-current={item.id === activeView ? "page" : undefined}
                onClick={() => onNavigate(item.id)}
              >
                <span className="nav-item-label">{item.label}</span>
                <span className="nav-item-hint">{item.hint}</span>
                {item.feature ? <span className="nav-item-feature">{item.feature}</span> : null}
              </button>
            ))}
          </nav>
        </div>

        <div />

        <div className="admin-reserved">
          Admin workspace is reserved for a later task. This shell leaves the slot out of the
          current production navigation.
        </div>
      </aside>

      <section className="studio-main">
        <header className="studio-topbar">
          <div className="studio-topbar-copy">
            <span className="studio-breadcrumb">Studio / {title}</span>
            <h1 className="studio-title">{title}</h1>
            <p className="studio-subtitle">{subtitle}</p>
          </div>
          {status ? <div className="studio-status">{status}</div> : null}
        </header>

        <nav className="mobile-nav" aria-label="Studio navigation mobile">
          {studioNavItems.map((item) => (
            <button
              key={item.id}
              type="button"
              className={item.id === activeView ? "mobile-nav-item active" : "mobile-nav-item"}
              aria-current={item.id === activeView ? "page" : undefined}
              onClick={() => onNavigate(item.id)}
            >
              <span className="mobile-nav-label">{item.label}</span>
              <span className="mobile-nav-hint">{item.hint}</span>
            </button>
          ))}
        </nav>

        <div className="studio-content">{children}</div>
      </section>
    </div>
  );
}
