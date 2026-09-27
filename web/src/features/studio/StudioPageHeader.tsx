import { useEffect, useRef, type ReactNode } from "react";

import { CONSOLE_PAGE_METADATA, type ConsolePage } from "./StudioNavigation";

const PAGE_MARKERS: Record<ConsolePage, string> = {
  overview: "01", onboarding: "01", activity: "04", device: "02", capture: "02",
  infer: "02", control: "03", models: "02", management: "06", license: "06",
  params: "03", "control-test": "02", latency: "02", settings: "05", about: "06",
};

export function StudioPageHeader({ page, action }: { page: ConsolePage; action?: ReactNode }) {
  const metadata = CONSOLE_PAGE_METADATA[page];
  const headingRef = useRef<HTMLHeadingElement>(null);
  const previousPageRef = useRef(page);

  useEffect(() => {
    if (previousPageRef.current === page) return;
    previousPageRef.current = page;
    headingRef.current?.focus({ preventScroll: true });
  }, [page]);

  return (
    <header className="console-page-header">
      <div className="console-page-marker" aria-hidden="true">
        <span>{PAGE_MARKERS[page]}</span>
        <i />
      </div>
      <div className="console-page-heading">
        <span>{metadata.group === "NovaSight" ? "NovaSight / Live system" : metadata.group}</span>
        <h1 ref={headingRef} tabIndex={-1}>{metadata.title}</h1>
        <p>{metadata.description}</p>
      </div>
      {action ? <div className="console-page-action">{action}</div> : null}
    </header>
  );
}
