import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

import mannequinTarget from "../../assets/aim-target/mannequin-target.webp";

export type AimRole = "head" | "body" | "other";
export type AimRoleRatios = Record<AimRole, number>;

const ROLE_META: Record<AimRole, { label: string; caption: string; zoneTop: number; zoneHeight: number }> = {
  head: { label: "头部", caption: "头部框内", zoneTop: 8, zoneHeight: 22 },
  body: { label: "身体", caption: "身体框内", zoneTop: 24, zoneHeight: 45 },
  other: { label: "其他", caption: "其他框内", zoneTop: 10, zoneHeight: 78 }
};

const ROLES: AimRole[] = ["head", "body", "other"];

function clampRatio(value: number): number {
  return Math.round(Math.max(0, Math.min(1, value)) * 100) / 100;
}

type AimTargetRangeProps = {
  disabled?: boolean;
  ratios: AimRoleRatios;
  onCommit: (role: AimRole, ratio: number) => void | Promise<void>;
};

export function AimTargetRange({ disabled = false, ratios, onCommit }: AimTargetRangeProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef<AimRole | null>(null);
  const [draft, setDraft] = useState(ratios);

  useEffect(() => {
    if (draggingRef.current === null) {
      setDraft(ratios);
    }
  }, [ratios]);

  const ratioFromPointer = (role: AimRole, clientY: number): number => {
    const rect = stageRef.current?.getBoundingClientRect();
    if (!rect || rect.height <= 0) {
      return draft[role];
    }
    const meta = ROLE_META[role];
    const pointerPercent = ((clientY - rect.top) / rect.height) * 100;
    return clampRatio((pointerPercent - meta.zoneTop) / meta.zoneHeight);
  };

  const updateDraft = (role: AimRole, ratio: number) => {
    setDraft((current) => ({ ...current, [role]: clampRatio(ratio) }));
  };

  const commit = (role: AimRole, ratio: number) => {
    const normalized = clampRatio(ratio);
    updateDraft(role, normalized);
    void onCommit(role, normalized);
  };

  const beginDrag = (role: AimRole, event: PointerEvent<HTMLButtonElement>) => {
    if (disabled) {
      return;
    }
    draggingRef.current = role;
    event.currentTarget.setPointerCapture(event.pointerId);
    updateDraft(role, ratioFromPointer(role, event.clientY));
  };

  const handleKey = (role: AimRole, event: KeyboardEvent<HTMLButtonElement>) => {
    if (disabled) {
      return;
    }
    const step = event.shiftKey ? 0.05 : 0.01;
    let next: number | null = null;
    if (event.key === "ArrowUp" || event.key === "ArrowLeft") next = draft[role] - step;
    if (event.key === "ArrowDown" || event.key === "ArrowRight") next = draft[role] + step;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = 1;
    if (next === null) {
      return;
    }
    event.preventDefault();
    commit(role, next);
  };

  return (
    <section className="aim-target-panel" aria-labelledby="aim-target-panel-title">
      <div className="aim-target-panel-heading">
        <div>
          <span className="class-config-eyebrow">AIM CALIBRATION RANGE</span>
          <h3 id="aim-target-panel-title">垂直瞄点靶场</h3>
        </div>
        <span className="aim-target-axis-lock">X 轴固定居中</span>
      </div>

      <div className="aim-target-stage" ref={stageRef}>
        <div className="aim-target-grid" aria-hidden="true" />
        <div className="aim-target-center-axis" aria-hidden="true" />
        <img alt="正面人物训练靶" className="aim-target-mannequin" draggable={false} src={mannequinTarget} />
        {ROLES.map((role) => {
          const meta = ROLE_META[role];
          const top = meta.zoneTop + draft[role] * meta.zoneHeight;
          return (
            <div className={`aim-role-guide aim-role-guide-${role}`} key={role} style={{ top: `${top}%` }}>
              <span className="aim-role-guide-label">
                <b>{meta.label}</b>
                <small>{Math.round(draft[role] * 100)}%</small>
              </span>
              <span className="aim-role-guide-line" aria-hidden="true" />
              <button
                aria-label={`${meta.label}瞄点，当前为框内 ${Math.round(draft[role] * 100)}%`}
                aria-orientation="vertical"
                aria-valuemax={100}
                aria-valuemin={0}
                aria-valuenow={Math.round(draft[role] * 100)}
                className="aim-role-guide-handle"
                disabled={disabled}
                onKeyDown={(event) => handleKey(role, event)}
                onPointerCancel={(event) => {
                  draggingRef.current = null;
                  event.currentTarget.releasePointerCapture(event.pointerId);
                  setDraft(ratios);
                }}
                onPointerDown={(event) => beginDrag(role, event)}
                onPointerMove={(event) => {
                  if (draggingRef.current === role) {
                    updateDraft(role, ratioFromPointer(role, event.clientY));
                  }
                }}
                onPointerUp={(event) => {
                  if (draggingRef.current === role) {
                    const next = ratioFromPointer(role, event.clientY);
                    draggingRef.current = null;
                    event.currentTarget.releasePointerCapture(event.pointerId);
                    commit(role, next);
                  }
                }}
                role="slider"
                type="button"
              />
            </div>
          );
        })}
        <span className="aim-target-stage-caption">拖动彩色标记，只调整各角色 bbox 内的 Y 比例</span>
      </div>

      <div className="aim-role-controls">
        {ROLES.map((role) => {
          const meta = ROLE_META[role];
          return (
            <label className={`aim-role-control aim-role-control-${role}`} key={role}>
              <span><i />{meta.label}<small>{meta.caption}</small></span>
              <span className="aim-role-control-input">
                <button disabled={disabled || draft[role] <= 0} onClick={() => commit(role, draft[role] - 0.01)} type="button">−</button>
                <input
                  aria-label={`${meta.label}框内垂直瞄点百分比`}
                  disabled={disabled}
                  max={100}
                  min={0}
                  onChange={(event) => updateDraft(role, Number(event.target.value) / 100)}
                  onBlur={() => commit(role, draft[role])}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") event.currentTarget.blur();
                  }}
                  step={1}
                  type="number"
                  value={Math.round(draft[role] * 100)}
                />
                <em>%</em>
                <button disabled={disabled || draft[role] >= 1} onClick={() => commit(role, draft[role] + 0.01)} type="button">+</button>
              </span>
            </label>
          );
        })}
      </div>
    </section>
  );
}
