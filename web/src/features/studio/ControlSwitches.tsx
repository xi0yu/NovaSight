import { useCallback, useEffect, useState } from "react";

export function TargetControlSwitch({
  enabled,
  pending,
  status,
  onToggle,
}: {
  enabled: boolean;
  pending: boolean;
  status: string;
  onToggle: (enabled: boolean) => void | boolean | Promise<void | boolean>;
}) {
  return (
    <button
      aria-busy={pending}
      aria-checked={enabled}
      aria-label={`目标控制，${status}`}
      className={`target-control-switch${enabled ? " on" : ""}`}
      disabled={pending}
      onClick={() => void Promise.resolve(onToggle(!enabled)).catch(() => undefined)}
      role="switch"
      type="button"
    >
      <span>
        <small>TARGET CONTROL</small>
        <b>{pending ? "正在处理" : status}</b>
      </span>
      <i aria-hidden="true"><span /></i>
    </button>
  );
}

export function ModuleSwitch({
  label,
  detail,
  compact = false,
  enabled,
  disabled = false,
  optimistic = true,
  onToggle
}: {
  label: string;
  detail?: string;
  compact?: boolean;
  enabled: boolean;
  disabled?: boolean;
  optimistic?: boolean;
  onToggle: (enabled: boolean) => Promise<void | boolean> | void | boolean;
}) {
  const [visualEnabled, setVisualEnabled] = useState(enabled);
  const [pending, setPending] = useState(false);

  useEffect(() => {
    if (!pending) {
      setVisualEnabled(enabled);
    }
  }, [enabled, pending]);

  const toggle = useCallback(async () => {
    if (pending || disabled) {
      return;
    }
    const next = !visualEnabled;
    if (optimistic) {
      setVisualEnabled(next);
    }
    setPending(true);
    try {
      const applied = await onToggle(next);
      if (!optimistic && applied !== false) {
        setVisualEnabled(next);
      }
    } catch {
      setVisualEnabled(enabled);
    } finally {
      setPending(false);
    }
  }, [disabled, enabled, onToggle, optimistic, pending, visualEnabled]);

  return (
    <button type="button"
      aria-busy={pending}
      aria-pressed={visualEnabled}
      className={`module-switch${visualEnabled ? " on" : ""}${compact ? " compact" : ""}`}
      onClick={() => void toggle()}
      disabled={pending || disabled}
    >
      <span>
        <b>{label}</b>
        {detail ? <small>{detail}</small> : null}
      </span>
      <i aria-hidden="true">{pending ? "处理中" : visualEnabled ? "开" : "关"}</i>
    </button>
  );
}
