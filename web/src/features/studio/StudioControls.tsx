import { useCallback, useEffect, useRef, useState } from "react";

import { formatNumberDraft, resolveNumberDraft } from "./numberDraft";

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
  const [draftValue, setDraftValue] = useState(value);
  const [draftText, setDraftText] = useState(() => formatNumberDraft(value, digits));
  const [isEditing, setIsEditing] = useState(false);
  const committingRef = useRef(false);
  const draftTextRef = useRef(formatNumberDraft(value, digits));

  useEffect(() => {
    if (!isEditing) {
      const nextText = formatNumberDraft(value, digits);
      draftTextRef.current = nextText;
      setDraftValue(value);
      setDraftText(nextText);
    }
  }, [digits, isEditing, value]);

  const commit = useCallback((candidateText = draftTextRef.current) => {
    if (committingRef.current) {
      return;
    }
    const next = resolveNumberDraft(candidateText, value, min, max, digits);
    const nextText = formatNumberDraft(next, digits);
    draftTextRef.current = nextText;
    setDraftValue(next);
    setDraftText(nextText);
    if (next !== Number(value.toFixed(digits))) {
      committingRef.current = true;
      void Promise.resolve(onCommit(next)).finally(() => {
        committingRef.current = false;
        setIsEditing(false);
      });
    } else {
      const valueText = formatNumberDraft(value, digits);
      draftTextRef.current = valueText;
      setDraftValue(value);
      setDraftText(valueText);
      setIsEditing(false);
    }
  }, [digits, max, min, onCommit, value]);

  return (
    <div className="console-row">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={draftValue}
        onBlur={() => commit()}
        onChange={(event) => {
          const next = clampNumber(Number(event.target.value), min, max);
          const nextText = formatNumberDraft(next, digits);
          setIsEditing(true);
          draftTextRef.current = nextText;
          setDraftValue(next);
          setDraftText(nextText);
        }}
        onFocus={() => setIsEditing(true)}
        onPointerDown={() => setIsEditing(true)}
        onPointerUp={() => commit()}
        onTouchEnd={() => commit()}
      />
      <input
        inputMode="decimal"
        type="text"
        value={draftText}
        onBlur={() => commit()}
        onFocus={() => setIsEditing(true)}
        onChange={(event) => {
          const nextText = event.target.value;
          const parsed = Number(nextText);
          setIsEditing(true);
          draftTextRef.current = nextText;
          setDraftText(nextText);
          if (nextText.trim() !== "" && Number.isFinite(parsed)) {
            const next = clampNumber(parsed, min, max);
            setDraftValue(next);
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
