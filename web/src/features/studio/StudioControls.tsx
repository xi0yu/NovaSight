import { useCallback, useEffect, useRef, useState } from "react";

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function NumberControl({
  label,
  detail,
  value,
  min,
  max,
  step,
  onCommit
}: {
  label: string;
  detail?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onCommit: (value: number) => Promise<void> | void;
}) {
  return (
    <>
      <label title={detail}>{label}</label>
      <CommitNumberControl
        value={value}
        min={min}
        max={max}
        step={step}
        digits={step >= 1 ? 0 : 2}
        onCommit={onCommit}
      />
    </>
  );
}

export function CommitNumberControl({
  value,
  min,
  max,
  step,
  digits,
  onCommit
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  digits: number;
  onCommit: (value: number) => Promise<void> | void;
}) {
  const [draft, setDraft] = useState(value);
  const [isEditing, setIsEditing] = useState(false);
  const committingRef = useRef(false);

  useEffect(() => {
    if (!isEditing) {
      setDraft(value);
    }
  }, [isEditing, value]);

  const commit = useCallback(() => {
    if (committingRef.current) {
      return;
    }
    const next = clampNumber(Number(draft.toFixed(digits)), min, max);
    if (Math.abs(next - value) >= step / 2) {
      committingRef.current = true;
      setDraft(next);
      void Promise.resolve(onCommit(next)).finally(() => {
        committingRef.current = false;
        setIsEditing(false);
      });
    } else {
      setDraft(value);
      setIsEditing(false);
    }
  }, [draft, digits, max, min, onCommit, step, value]);

  return (
    <div className="console-row">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={draft}
        onBlur={commit}
        onChange={(event) => {
          setIsEditing(true);
          setDraft(clampNumber(Number(event.target.value), min, max));
        }}
        onFocus={() => setIsEditing(true)}
        onPointerDown={() => setIsEditing(true)}
        onPointerUp={commit}
        onTouchEnd={commit}
      />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={Number.isInteger(draft) ? String(draft) : draft.toFixed(digits)}
        onBlur={commit}
        onFocus={() => setIsEditing(true)}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) {
            setIsEditing(true);
            setDraft(clampNumber(next, min, max));
          }
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
        }}
      />
    </div>
  );
}

export function TextControl({
  label,
  value,
  onCommit
}: {
  label: string;
  value: string;
  onCommit: (value: string) => Promise<void> | void;
}) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  const commit = useCallback(() => {
    const next = draft.trim();
    if (next !== value) {
      void onCommit(next);
    }
  }, [draft, onCommit, value]);

  return (
    <>
      <label>{label}</label>
      <input
        value={draft}
        onBlur={commit}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()}
      />
    </>
  );
}

export function InlineTextControl({
  value,
  placeholder,
  ariaLabel,
  onCommit
}: {
  value: string;
  placeholder: string;
  ariaLabel: string;
  onCommit: (value: string) => Promise<void> | void;
}) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  const commit = useCallback(() => {
    const next = draft.trim();
    if (next !== value) {
      void onCommit(next);
    }
  }, [draft, onCommit, value]);

  return (
    <input
      aria-label={ariaLabel}
      value={draft}
      placeholder={placeholder}
      onBlur={commit}
      onChange={(event) => setDraft(event.target.value)}
      onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()}
    />
  );
}

export function ClassAimRatioControl({
  classId,
  defaultRatio,
  overrideRatio,
  onCommit
}: {
  classId: number;
  defaultRatio: number;
  overrideRatio: number | undefined;
  onCommit: (value: number | null) => Promise<void> | void;
}) {
  const custom = overrideRatio !== undefined;
  const effectiveRatio = custom ? overrideRatio : defaultRatio;
  const [draft, setDraft] = useState(effectiveRatio);
  useEffect(() => setDraft(effectiveRatio), [effectiveRatio]);

  const commit = useCallback(() => {
    if (!custom) {
      return;
    }
    const next = clampNumber(Number(draft.toFixed(2)), 0, 1);
    setDraft(next);
    if (next !== overrideRatio) {
      void onCommit(next);
    }
  }, [custom, draft, onCommit, overrideRatio]);

  return (
    <div className="class-aim-control">
      <button
        className={custom ? "class-aim-mode custom" : "class-aim-mode"}
        type="button"
        onClick={() => void onCommit(custom ? null : defaultRatio)}
      >
        {custom ? "单独设置" : "使用默认"}
      </button>
      <input
        aria-label={`cls ${classId} 瞄点高度`}
        disabled={!custom}
        min={0}
        max={1}
        step={0.01}
        type="number"
        value={draft.toFixed(2)}
        onBlur={commit}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) {
            setDraft(clampNumber(next, 0, 1));
          }
        }}
        onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()}
      />
      <div className="class-aim-preview" aria-hidden="true">
        <i style={{ top: `${effectiveRatio * 100}%` }} />
      </div>
    </div>
  );
}
