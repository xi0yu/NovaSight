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
    <div className="number-control-field">
      <label title={detail}>{label}</label>
      <CommitNumberControl
        value={value}
        min={min}
        max={max}
        step={step}
        digits={step >= 1 ? 0 : 2}
        onCommit={onCommit}
      />
    </div>
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
    <div className={custom ? "class-aim-control custom" : "class-aim-control inherited"}>
      <div className="class-aim-mode-group" role="group" aria-label={`cls ${classId} 垂直瞄点模式`}>
        <button
          aria-pressed={!custom}
          className={!custom ? "active" : ""}
          type="button"
          onClick={() => custom && void onCommit(null)}
        >
          跟随默认
        </button>
        <button
          aria-pressed={custom}
          className={custom ? "active" : ""}
          type="button"
          onClick={() => !custom && void onCommit(defaultRatio)}
        >
          独立设置
        </button>
      </div>
      <label className="class-aim-value">
        <span className="visually-hidden">cls {classId} 垂直瞄点百分比</span>
        <input
          aria-label={`cls ${classId} 垂直瞄点百分比`}
          disabled={!custom}
          min={0}
          max={100}
          step={1}
          type="number"
          value={Math.round(draft * 100)}
          onBlur={commit}
          onChange={(event) => {
            const next = Number(event.target.value);
            if (Number.isFinite(next)) {
              setDraft(clampNumber(next / 100, 0, 1));
            }
          }}
          onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()}
        />
        <span aria-hidden="true">%</span>
      </label>
      <div className="class-aim-preview" aria-hidden="true">
        <i style={{ top: `${effectiveRatio * 100}%` }} />
      </div>
    </div>
  );
}
