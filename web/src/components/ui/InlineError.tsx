type InlineErrorProps = {
  message?: string;
};

export function InlineError({ message }: InlineErrorProps) {
  if (!message) {
    return null;
  }

  return (
    <div className="inline-error" role="status">
      <strong>请求失败</strong>
      <span>{message}</span>
    </div>
  );
}
