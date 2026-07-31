import { useCallback, useEffect, useId, useRef, useState, type CSSProperties } from "react";

import { formatNumberDraft, resolveNumberDraft } from "./numberDraft";

export type ParameterApplyMode = "live" | "save" | "launch" | "restart";
export type ParameterRiskLevel = "normal" | "advanced" | "calibration";
export type ParameterNumberKind = "slider" | "stepper";
export type ParameterSelectOption = {
  value: string;
  label: string;
  disabled?: boolean;
};

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function decimalPlacesForStep(step: number): number {
  if (!Number.isFinite(step) || step <= 0 || Number.isInteger(step)) {
    return 0;
  }
  const scientificMatch = step.toString().match(/e-(\d+)$/i);
  if (scientificMatch) {
    return Number(scientificMatch[1]);
  }
  return Math.min(8, step.toString().split(".")[1]?.length ?? 0);
}

function percentWithin(value: number, min: number, max: number): number {
  if (!Number.isFinite(value) || max <= min) {
    return 0;
  }
  return clampNumber(((value - min) / (max - min)) * 100, 0, 100);
}

function SliderNumberControl({
  value,
  min,
  max,
  rangeMin = min,
  rangeMax = max,
  rangeInputId,
  textInputId,
  step,
  digits,
  ariaLabel,
  disabled = false,
  onDraftChange,
  onCommit
}: {
  value: number;
  min: number;
  max: number;
  rangeMin?: number;
  rangeMax?: number;
  rangeInputId?: string;
  textInputId?: string;
  step: number;
  digits: number;
  ariaLabel?: string;
  disabled?: boolean;
  onDraftChange?: (value: number) => void;
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
    if (disabled) {
      return;
    }
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
  }, [digits, disabled, max, min, onCommit, value]);

  const preferredMin = clampNumber(Math.min(rangeMin, rangeMax), min, max);
  const preferredMax = clampNumber(Math.max(rangeMin, rangeMax), preferredMin, max);
  const sliderMin = clampNumber(Math.min(preferredMin, value), min, max);
  const sliderMax = clampNumber(Math.max(preferredMax, value, sliderMin), sliderMin, max);
  const sliderValue = clampNumber(draftValue, sliderMin, sliderMax);
  const rangeStyle = {
    "--parameter-range-value": `${percentWithin(sliderValue, sliderMin, sliderMax)}%`,
    "--parameter-range-recommended-start": `${percentWithin(preferredMin, sliderMin, sliderMax)}%`,
    "--parameter-range-recommended-end": `${percentWithin(preferredMax, sliderMin, sliderMax)}%`
  } as CSSProperties;

  return (
    <div className="console-row">
      <input
        aria-label={ariaLabel ? `${ariaLabel} 滑块` : undefined}
        disabled={disabled}
        id={rangeInputId}
        type="range"
        min={sliderMin}
        max={sliderMax}
        step={step}
        style={rangeStyle}
        value={sliderValue}
        onBlur={() => commit()}
        onChange={(event) => {
          const next = clampNumber(Number(event.target.value), min, max);
          const nextText = formatNumberDraft(next, digits);
          setIsEditing(true);
          draftTextRef.current = nextText;
          setDraftValue(next);
          setDraftText(nextText);
          onDraftChange?.(next);
        }}
        onFocus={() => setIsEditing(true)}
        onPointerDown={() => setIsEditing(true)}
        onPointerUp={() => commit()}
        onTouchEnd={() => commit()}
      />
      <input
        aria-label={ariaLabel ? `${ariaLabel} 数值` : undefined}
        disabled={disabled}
        id={textInputId}
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
            onDraftChange?.(next);
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

function StepperNumberControl({
  value,
  min,
  max,
  inputId,
  step,
  digits,
  ariaLabel,
  disabled = false,
  onDraftChange,
  onCommit
}: {
  value: number;
  min: number;
  max: number;
  inputId?: string;
  step: number;
  digits: number;
  ariaLabel?: string;
  disabled?: boolean;
  onDraftChange?: (value: number) => void;
  onCommit: (value: number) => Promise<void> | void;
}) {
  const [draftText, setDraftText] = useState(() => formatNumberDraft(value, digits));
  const [isEditing, setIsEditing] = useState(false);
  const committingRef = useRef(false);
  const draftTextRef = useRef(formatNumberDraft(value, digits));

  useEffect(() => {
    if (!isEditing) {
      const nextText = formatNumberDraft(value, digits);
      draftTextRef.current = nextText;
      setDraftText(nextText);
    }
  }, [digits, isEditing, value]);

  const commit = useCallback((candidateText = draftTextRef.current) => {
    if (disabled) {
      return;
    }
    if (committingRef.current) {
      return;
    }
    const next = resolveNumberDraft(candidateText, value, min, max, digits);
    const nextText = formatNumberDraft(next, digits);
    draftTextRef.current = nextText;
    setDraftText(nextText);
    if (next !== Number(value.toFixed(digits))) {
      committingRef.current = true;
      void Promise.resolve(onCommit(next)).finally(() => {
        committingRef.current = false;
        setIsEditing(false);
      });
    } else {
      setIsEditing(false);
    }
  }, [digits, disabled, max, min, onCommit, value]);

  const stepBy = useCallback((direction: -1 | 1) => {
    if (disabled) {
      return;
    }
    const next = clampNumber(Number((value + direction * step).toFixed(digits)), min, max);
    draftTextRef.current = formatNumberDraft(next, digits);
    setDraftText(draftTextRef.current);
    onDraftChange?.(next);
    commit(draftTextRef.current);
  }, [commit, digits, disabled, max, min, onDraftChange, step, value]);

  return (
    <div className="parameter-stepper-control">
      <button
        aria-label="减少数值"
        className="parameter-stepper-button"
        disabled={disabled || value <= min}
        onClick={() => stepBy(-1)}
        type="button"
      >
        -
      </button>
      <input
        aria-label={ariaLabel ? `${ariaLabel} 数值` : undefined}
        disabled={disabled}
        id={inputId}
        inputMode="decimal"
        type="text"
        value={draftText}
        onBlur={() => commit()}
        onFocus={() => setIsEditing(true)}
        onChange={(event) => {
          draftTextRef.current = event.target.value;
          setDraftText(event.target.value);
          setIsEditing(true);
          const parsed = Number(event.target.value);
          if (event.target.value.trim() !== "" && Number.isFinite(parsed)) {
            onDraftChange?.(clampNumber(parsed, min, max));
          }
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
        }}
      />
      <button
        aria-label="增加数值"
        className="parameter-stepper-button"
        disabled={disabled || value >= max}
        onClick={() => stepBy(1)}
        type="button"
      >
        +
      </button>
    </div>
  );
}

function applyModeLabel(applyMode: ParameterApplyMode): string {
  if (applyMode === "live") {
    return "即时";
  }
  if (applyMode === "restart") {
    return "需重启";
  }
  if (applyMode === "launch") {
    return "启动时";
  }
  return "保存后";
}

function rangeLabel(min: number, max: number, unit = ""): string {
  const suffix = unit ? ` ${unit}` : "";
  return `${min}${suffix} - ${max}${suffix}`;
}

export function ParameterNumberControl({
  label,
  detail,
  value,
  min,
  max,
  recommendedMin = min,
  recommendedMax = max,
  step,
  unit,
  kind = "slider",
  disabled = false,
  applyMode = "save",
  riskLevel = "normal",
  onDraftChange,
  onCommit
}: {
  label: string;
  detail?: string;
  value: number;
  min: number;
  max: number;
  recommendedMin?: number;
  recommendedMax?: number;
  step: number;
  unit?: string;
  kind?: ParameterNumberKind;
  disabled?: boolean;
  applyMode?: ParameterApplyMode;
  riskLevel?: ParameterRiskLevel;
  onDraftChange?: (value: number) => void;
  onCommit: (value: number) => Promise<void> | void;
}) {
  const controlId = useId();
  const digits = decimalPlacesForStep(step);
  const normalizedRecommendedMin = clampNumber(Math.min(recommendedMin, recommendedMax), min, max);
  const normalizedRecommendedMax = clampNumber(Math.max(recommendedMin, recommendedMax), normalizedRecommendedMin, max);
  const hasRecommendedRange = normalizedRecommendedMin !== min || normalizedRecommendedMax !== max;
  const outsideRecommendedRange = hasRecommendedRange && (value < normalizedRecommendedMin || value > normalizedRecommendedMax);

  return (
    <div className={`parameter-control-field parameter-control-${kind} parameter-risk-${riskLevel}`}>
      <div className="parameter-control-header">
        <label htmlFor={`${controlId}-${kind === "stepper" ? "value" : "range"}`} title={detail}>{label}</label>
        <span className="parameter-control-meta" aria-label="参数属性">
          {unit ? <span>{unit}</span> : null}
          <span>{applyModeLabel(applyMode)}</span>
          {outsideRecommendedRange ? <span data-tone="warning">超推荐</span> : null}
          {riskLevel !== "normal" ? <span>{riskLevel === "calibration" ? "标定" : "高级"}</span> : null}
        </span>
      </div>
      {detail ? <p className="parameter-control-detail">{detail}</p> : null}
      {kind === "stepper" ? (
        <StepperNumberControl
          value={value}
          min={min}
          max={max}
          step={step}
          digits={digits}
          inputId={`${controlId}-value`}
          ariaLabel={label}
          disabled={disabled}
          onDraftChange={onDraftChange}
          onCommit={onCommit}
        />
      ) : (
        <SliderNumberControl
          value={value}
          min={min}
          max={max}
          rangeMin={normalizedRecommendedMin}
          rangeMax={normalizedRecommendedMax}
          rangeInputId={`${controlId}-range`}
          textInputId={`${controlId}-value`}
          step={step}
          digits={digits}
          ariaLabel={label}
          disabled={disabled}
          onDraftChange={onDraftChange}
          onCommit={onCommit}
        />
      )}
      {hasRecommendedRange ? (
        <div className="parameter-control-range">
          <span>建议 {rangeLabel(normalizedRecommendedMin, normalizedRecommendedMax, unit)}</span>
          {outsideRecommendedRange ? <span>边界 {rangeLabel(min, max, unit)}</span> : null}
        </div>
      ) : null}
    </div>
  );
}

export function ParameterPresetControl({
  label,
  detail,
  density = "normal",
  options
}: {
  label: string;
  detail?: string;
  density?: "normal" | "compact";
  options: Array<{
    id: string;
    label: string;
    detail: string;
    active?: boolean;
    disabled?: boolean;
    onSelect: () => Promise<void> | void;
  }>;
}) {
  return (
    <section className={density === "compact" ? "parameter-preset-control compact" : "parameter-preset-control"} aria-label={label}>
      <header>
        <b>{label}</b>
        {detail ? <span>{detail}</span> : null}
      </header>
      <div className="parameter-preset-options">
        {options.map((option) => (
          <button
            aria-pressed={option.active === true}
            className={option.active ? "parameter-preset-option active" : "parameter-preset-option"}
            disabled={option.disabled}
            key={option.id}
            onClick={() => void option.onSelect()}
            type="button"
          >
            <b>{option.label}</b>
            <small>{option.detail}</small>
          </button>
        ))}
      </div>
    </section>
  );
}

export function TextControl({
  label,
  detail,
  value,
  placeholder,
  disabled = false,
  applyMode = "save",
  riskLevel = "normal",
  onDraftChange,
  onEnter,
  onCommit
}: {
  label: string;
  detail?: string;
  value: string;
  placeholder?: string;
  disabled?: boolean;
  applyMode?: ParameterApplyMode;
  riskLevel?: ParameterRiskLevel;
  onDraftChange?: (value: string) => void;
  onEnter?: () => Promise<void> | void;
  onCommit: (value: string) => Promise<void> | void;
}) {
  const controlId = useId();
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  const commit = useCallback(() => {
    const next = draft.trim();
    if (next !== value) {
      void onCommit(next);
    }
  }, [draft, onCommit, value]);

  return (
    <div className={`parameter-control-field parameter-control-text parameter-risk-${riskLevel}`}>
      <div className="parameter-control-header">
        <label htmlFor={controlId} title={detail}>{label}</label>
        <span className="parameter-control-meta" aria-label="参数属性">
          <span>{applyModeLabel(applyMode)}</span>
          {riskLevel !== "normal" ? <span>{riskLevel === "calibration" ? "标定" : "高级"}</span> : null}
        </span>
      </div>
      {detail ? <p className="parameter-control-detail">{detail}</p> : null}
      <input
        id={controlId}
        disabled={disabled}
        value={draft}
        placeholder={placeholder}
        onBlur={commit}
        onChange={(event) => {
          setDraft(event.target.value);
          onDraftChange?.(event.target.value);
        }}
        onKeyDown={(event) => {
          if (event.key !== "Enter") {
            return;
          }
          event.preventDefault();
          event.currentTarget.blur();
          if (onEnter) {
            void onEnter();
          }
        }}
      />
    </div>
  );
}

export function SelectControl({
  label,
  detail,
  value,
  options,
  disabled = false,
  applyMode = "save",
  riskLevel = "normal",
  onCommit
}: {
  label: string;
  detail?: string;
  value: string;
  options: ParameterSelectOption[];
  disabled?: boolean;
  applyMode?: ParameterApplyMode;
  riskLevel?: ParameterRiskLevel;
  onCommit: (value: string) => Promise<void> | void;
}) {
  const controlId = useId();

  return (
    <div className={`parameter-control-field parameter-control-select parameter-risk-${riskLevel}`}>
      <div className="parameter-control-header">
        <label htmlFor={controlId} title={detail}>{label}</label>
        <span className="parameter-control-meta" aria-label="参数属性">
          <span>{applyModeLabel(applyMode)}</span>
          {riskLevel !== "normal" ? <span>{riskLevel === "calibration" ? "标定" : "高级"}</span> : null}
        </span>
      </div>
      {detail ? <p className="parameter-control-detail">{detail}</p> : null}
      <select
        disabled={disabled}
        id={controlId}
        value={value}
        onChange={(event) => void onCommit(event.target.value)}
      >
        {options.map((option) => (
          <option disabled={option.disabled} key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export function InlineNumberControl({
  value,
  ariaLabel,
  disabled = false,
  digits = 0,
  onCommit
}: {
  value: number;
  ariaLabel: string;
  disabled?: boolean;
  digits?: number;
  onCommit: (value: number) => Promise<void> | void;
}) {
  const [draft, setDraft] = useState(() => formatNumberDraft(value, digits));
  useEffect(() => setDraft(formatNumberDraft(value, digits)), [digits, value]);
  const commit = useCallback(() => {
    const parsed = Number(draft);
    if (!Number.isFinite(parsed)) {
      setDraft(formatNumberDraft(value, digits));
      return;
    }
    const next = digits === 0 ? Math.round(parsed) : Number(parsed.toFixed(digits));
    setDraft(formatNumberDraft(next, digits));
    if (next !== value) {
      void onCommit(next);
    }
  }, [digits, draft, onCommit, value]);

  return (
    <input
      aria-label={ariaLabel}
      className="inline-number-control"
      disabled={disabled}
      inputMode="decimal"
      type="text"
      value={draft}
      onBlur={commit}
      onChange={(event) => setDraft(event.target.value)}
      onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()}
    />
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
