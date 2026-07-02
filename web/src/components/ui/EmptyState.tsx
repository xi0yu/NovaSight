type EmptyStateProps = {
  title: string;
  detail: string;
  command?: string;
};

export function EmptyState({ title, detail, command }: EmptyStateProps) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <span>{detail}</span>
      {command ? <code>{command}</code> : null}
    </div>
  );
}
