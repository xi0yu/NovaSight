import { useEffect, useRef, type ReactNode } from "react";

import { CONSOLE_PAGE_METADATA, type ConsolePage } from "./StudioNavigation";

export function StudioPageHeader({ page, action, visuallyHidden = false }: { page: ConsolePage; action?: ReactNode; visuallyHidden?: boolean }) {
  const metadata = CONSOLE_PAGE_METADATA[page];
  const headingRef = useRef<HTMLHeadingElement>(null);
  const previousPageRef = useRef(page);

  useEffect(() => {
    if (previousPageRef.current === page) return;
    previousPageRef.current = page;
    headingRef.current?.focus({ preventScroll: true });
  }, [page]);

  return (
    <header className={visuallyHidden ? "console-page-header visually-hidden" : "console-page-header"}>
      <div className="console-page-heading">
        <span>{metadata.group}</span>
        <h1 ref={headingRef} tabIndex={-1}>{metadata.title}</h1>
        <p>{metadata.description}</p>
      </div>
      {action ? <div className="console-page-action">{action}</div> : null}
    </header>
  );
}
