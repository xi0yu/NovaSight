export function downloadJson(value: unknown, fileName: string): void {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.append(link);
  try { link.click(); }
  finally {
    link.remove();
    // Give the browser time to consume the download before releasing the blob.
    window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
  }
}
