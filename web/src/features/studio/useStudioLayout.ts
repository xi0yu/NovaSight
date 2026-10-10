import { useEffect, useState } from "react";
import { reportError } from "../../lib/toast";

import {
  DEFAULT_STUDIO_LAYOUT,
  decodeStudioLayout,
  type StudioLayout,
} from "../../contracts/studioLayout";

const REFRESH_INTERVAL_MS = 30_000;

export function useStudioLayout(): StudioLayout {
  const [layout, setLayout] = useState(DEFAULT_STUDIO_LAYOUT);

  useEffect(() => {
    let active = true;
    let lastError = "";
    const refresh = async () => {
      try {
        const response = await fetch("/studio-layout.json", { cache: "no-store" });
        if (!response.ok) throw new Error(`页面配置读取失败（${response.status}），继续使用已验证的布局。`);
        const next = decodeStudioLayout(await response.json());
        if (active) setLayout((current) => JSON.stringify(current) === JSON.stringify(next) ? current : next);
        lastError = "";
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (active && lastError !== message) {
          lastError = message;
          reportError(error, { source: "studio-layout", title: "页面配置未生效", popup: false });
        }
      }
    };
    void refresh();
    const interval = window.setInterval(() => void refresh(), REFRESH_INTERVAL_MS);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  return layout;
}
