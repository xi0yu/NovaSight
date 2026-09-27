import { useEffect, useState } from "react";

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
    const refresh = async () => {
      try {
        const response = await fetch("/studio-layout.json", { cache: "no-store" });
        if (!response.ok) return;
        const next = decodeStudioLayout(await response.json());
        if (active) setLayout(next);
      } catch {
        // Keep the last verified layout. Invalid remote configuration must not blank the Studio.
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
