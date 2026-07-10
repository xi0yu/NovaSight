import type { ReactNode } from "react";

import { NovaIcon } from "../components/visual";
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
      <header className="studio-topbar">
        <div className="studio-brand">
          <span className="brand-mark" aria-hidden="true">
            <NovaIcon name="prediction-line" size={22} strokeWidth={1.9} />
          </span>
          <div className="studio-brand-copy">
            <strong>NovaSight Studio</strong>
            <span>视觉控制工作台</span>
          </div>
        </div>

        <div className="studio-topbar-copy">
          <span className="studio-breadcrumb">项目：默认项目 / {title}</span>
          <h1 className="studio-title">{title}</h1>
          <p className="studio-subtitle">{subtitle}</p>
        </div>
        {status ? <div className="studio-status">{status}</div> : null}
      </header>

      <aside className="studio-sidebar">
        <div className="studio-sidebar-section">
          <nav className="studio-nav" aria-label="工作台导航">
            {studioNavItems.map((item) => (
              <button
                key={item.id}
                type="button"
                className={item.id === activeView ? "nav-item active" : "nav-item"}
                aria-current={item.id === activeView ? "page" : undefined}
                onClick={() => onNavigate(item.id)}
              >
                <span className="nav-item-icon">
                  <NovaIcon name={item.icon} size={20} />
                </span>
                <span className="nav-item-label">{item.label}</span>
                <span className="nav-item-hint">{item.hint}</span>
                {item.feature ? <span className="nav-item-feature">{item.feature}</span> : null}
              </button>
            ))}
          </nav>
        </div>

        <div />

        <div className="admin-reserved">
          当前版本聚焦采集、推理、模型和授权。管理区会在硬件控制闭环稳定后接入。
        </div>
      </aside>

      <section className="studio-main">
        <nav className="mobile-nav" aria-label="移动端工作台导航">
          {studioNavItems.map((item) => (
            <button
              key={item.id}
              type="button"
              className={item.id === activeView ? "mobile-nav-item active" : "mobile-nav-item"}
              aria-current={item.id === activeView ? "page" : undefined}
              onClick={() => onNavigate(item.id)}
            >
              <NovaIcon name={item.icon} size={16} />
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
