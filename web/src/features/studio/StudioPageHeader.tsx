import { CONSOLE_PAGE_METADATA, type ConsolePage } from "./StudioNavigation";

export function StudioPageHeader({ page }: { page: ConsolePage }) {
  const metadata = CONSOLE_PAGE_METADATA[page];
  return (
    <header className="console-page-header">
      <div className="console-page-heading">
        <span>{metadata.group}</span>
        <h1>{metadata.title}</h1>
        <p>{metadata.description}</p>
      </div>
    </header>
  );
}
