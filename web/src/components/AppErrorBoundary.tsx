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
      publicDetail: info.componentStack || error.message,
      popup: false,
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
        <strong>这个页面暂时无法显示</strong>
        <small>可以先重试打开页面。重新载入会丢失尚未保存的编辑；设备运行不会因此自动停止。</small>
        <details><summary>查看错误详情</summary><pre>{error.message || "发生未知渲染异常"}</pre></details>
        <div className="app-error-actions">
          <button className="console-button primary" type="button" onClick={this.retryRender}>重试打开</button>
          <button className="console-button" type="button" onClick={() => window.location.reload()}>重新载入</button>
        </div>
      </main>
    );
  }
}
