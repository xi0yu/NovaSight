import type { ReactNode } from "react";
import { NovaIcon } from "./visual";
import "./startup-screen.css";

export function StartupScreen({ message = "正在打开你的工作台…", children }: {
  message?: string;
  children?: ReactNode;
}) {
  return <main className="startup-screen" role="status" aria-label="正在打开 NovaSight">
    <div className="startup-screen-content">
      <span className="startup-screen-mark" aria-hidden="true"><NovaIcon name="prediction-line" size={36} /></span>
      <h2>NovaSight Studio</h2><p>{message}</p>
      <span className="startup-screen-progress" aria-hidden="true" />
      {children}
    </div>
  </main>;
}
