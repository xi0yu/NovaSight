type InlineErrorProps = {
  message?: string;
  title?: string;
};

export function InlineError({ message, title = "请求失败" }: InlineErrorProps) {
  if (!message) {
    return null;
  }

  return (
    <div className="inline-error" role="status">
      <strong>{title}</strong>
      <span>{message}</span>
    </div>
  );
}
