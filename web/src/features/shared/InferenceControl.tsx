export function InferenceControl({
  running,
  busy,
  onCommand
}: {
  running: boolean;
  busy: boolean;
  onCommand: (action: "start" | "stop") => void;
}) {
  return (
    <button
      className={running ? "button danger-button" : "button primary-button"}
      disabled={busy}
      onClick={() => onCommand(running ? "stop" : "start")}
      type="button"
    >
      {busy ? "处理中" : running ? "停止推理控制" : "启动推理控制"}
    </button>
  );
}
