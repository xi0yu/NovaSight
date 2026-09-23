import { Component, type ErrorInfo, type ReactNode } from "react";

import { reportError } from "../lib/toast";

interface AppErrorBoundaryProps {
  children: ReactNode;
}

interface AppErrorBoundaryState {
  error: Error | null;
}

export class AppErrorBoundary extends Component<
  AppErrorBoundaryProps,
  AppErrorBoundaryState
> {
  state: AppErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    reportError(error, {
      source: "react-render",
      title: "界面渲染失败",
      publicDetail: info.componentStack || error.message
    });
  }

  private retryRender = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <main className="route-loading-shell" role="alert">
        <span className="route-loading-mark" aria-hidden="true" />
        <strong>界面暂时无法渲染</strong>
        <small>{error.message || "发生未知渲染异常"}</small>
        <div className="app-error-actions">
          <button type="button" onClick={this.retryRender}>重试渲染</button>
          <button type="button" onClick={() => window.location.reload()}>重新载入</button>
        </div>
      </main>
    );
  }
}
