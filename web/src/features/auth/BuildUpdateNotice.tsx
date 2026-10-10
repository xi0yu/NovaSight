import { useEffect, useState } from "react";
import { Button } from "../../components/ui";
import { ActionConfirmationDialog } from "../studio/ActionConfirmationDialog";

export function BuildUpdateNotice() {
  const [available, setAvailable] = useState(false);
  const [confirming, setConfirming] = useState(false);
  useEffect(() => {
    if (import.meta.env.DEV) return;
    const controller = new AbortController();
    const check = async () => {
      try {
        const response = await fetch("/version.json", { cache: "no-store", signal: controller.signal });
        if (!response.ok) return;
        const data: unknown = await response.json();
        if (typeof data === "object" && data !== null && "version" in data && typeof data.version === "string") {
          setAvailable(data.version !== __NOVASIGHT_BUILD_VERSION__);
        }
      } catch { /* Version checks never change authentication or connection state. */ }
    };
    void check();
    const timer = window.setInterval(() => void check(), 60_000);
    const wake = () => void check();
    window.addEventListener("focus", wake);
    return () => { controller.abort(); window.clearInterval(timer); window.removeEventListener("focus", wake); };
  }, []);
  const refresh = () => {
    const event = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(event);
    if (event.defaultPrevented) setConfirming(true);
    else window.location.reload();
  };
  if (!available) return null;
  return <>
    <aside className="auth-recovery-notice" role="status"><span>新版本可用</span><Button variant="ghost" size="compact" onClick={refresh}>更新页面</Button></aside>
    {confirming ? <ActionConfirmationDialog busy={false} request={{ eyebrow: "页面更新", title: "还有未保存内容", description: "先完成保存再更新，当前输入会继续保留。", confirmLabel: "返回保存", onConfirm: () => setConfirming(false) }} onCancel={() => setConfirming(false)} onConfirm={() => setConfirming(false)} /> : null}
  </>;
}
