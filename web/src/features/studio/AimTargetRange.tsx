import { memo, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

import mannequinTarget from "../../assets/aim-target/mannequin-target-v2.webp";
import { ParameterNumberControl } from "./StudioControls";

export type AimRole = "head" | "body" | "other";
export type AimRoleRatios = Record<AimRole, number>;

const ROLE_META: Record<AimRole, { label: string; caption: string; zoneTop: number; zoneHeight: number }> = {
  // The mannequin is a semantic calibration surface, not one shared 0-100% ruler.
  // A head bbox spans only the head. Body/other bboxes commonly cover the whole
  // person, so their full range intentionally includes both head and torso.
  head: { label: "头部", caption: "0–100%：头皮到下巴", zoneTop: 0, zoneHeight: 16 },
  body: { label: "身体", caption: "0–100%：头皮到脚底", zoneTop: 0, zoneHeight: 100 },
  other: { label: "其他", caption: "0–100%：头皮到脚底", zoneTop: 0, zoneHeight: 100 }
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

export const AimTargetRange = memo(function AimTargetRange({ disabled = false, ratios, onCommit }: AimTargetRangeProps) {
  const roleRangeRefs = useRef<Partial<Record<AimRole, HTMLDivElement | null>>>({});
  const draggingRef = useRef<AimRole | null>(null);
  const [draft, setDraft] = useState(ratios);
  // Mirror of the latest draft so the keyDown handler can read fresh values
  // on rapid keypresses without waiting for React to commit a re-render.
  // Without this, holding Arrow reads `draft[role]` from a stale closure
  // and the slider visibly jumps back to the previous value between presses.
  const draftRef = useRef(draft);
  draftRef.current = draft;

  useEffect(() => {
    if (draggingRef.current === null) {
      setDraft(ratios);
    }
  }, [ratios]);

  const ratioFromPointer = (role: AimRole, clientY: number): number => {
    const rect = roleRangeRefs.current[role]?.getBoundingClientRect();
    if (!rect || rect.height <= 0) {
      return draftRef.current[role];
    }
    return clampRatio((clientY - rect.top) / rect.height);
  };

  // Per-role display value. While editing (dragging or focused) the local
  // draft wins; otherwise the prop value wins so a partial frame updates
  // the visual immediately, without waiting for the setDraft effect to
  // re-sync `draft` from the new prop. Keeps the guide and slider thumb
  // in lockstep during the brief render between prop change and effect run.
  const displayRatio = (role: AimRole): number =>
    draggingRef.current === role ? draft[role] : ratios[role];

  const updateDraft = (role: AimRole, ratio: number) => {
    setDraft((current) => {
      const next = { ...current, [role]: clampRatio(ratio) };
      draftRef.current = next;
      return next;
    });
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
    // Read the current value from the ref, not the closure-captured `draft`,
    // so rapid Arrow presses always advance from the latest committed value.
    const current = draftRef.current[role];
    let next: number | null = null;
    if (event.key === "ArrowUp" || event.key === "ArrowLeft") next = current - step;
    if (event.key === "ArrowDown" || event.key === "ArrowRight") next = current + step;
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
          <span className="class-config-eyebrow">瞄点标定</span>
          <h3 id="aim-target-panel-title">垂直瞄点靶场</h3>
        </div>
        <span className="aim-target-axis-lock">X 轴固定居中</span>
      </div>

      <div className="aim-target-stage">
        <div className="aim-target-grid" aria-hidden="true" />
        <div className="aim-target-figure">
          <div className="aim-target-center-axis" aria-hidden="true" />
          <img
            alt="正面人物训练靶"
            className="aim-target-mannequin"
            decoding="async"
            draggable={false}
            height={1737}
            loading="lazy"
            src={mannequinTarget}
            width={864}
          />
          {ROLES.map((role) => {
            const meta = ROLE_META[role];
            return (
              <div
                className={`aim-role-range aim-role-range-${role}`}
                key={role}
                ref={(node) => {
                  roleRangeRefs.current[role] = node;
                }}
                style={{ top: `${meta.zoneTop}%`, height: `${meta.zoneHeight}%` }}
              >
                <div className={`aim-role-guide aim-role-guide-${role}`} style={{ top: `${displayRatio(role) * 100}%` }}>
                  <span className="aim-role-guide-leading">
                    <span className="aim-role-guide-label">
                      <b>{meta.label}</b>
                      <small>{Math.round(displayRatio(role) * 100)}%</small>
                    </span>
                    <span className="aim-role-guide-line before" aria-hidden="true" />
                  </span>
                  <button
                    aria-label={`${meta.label}瞄点，当前为框内 ${Math.round(displayRatio(role) * 100)}%`}
                    aria-orientation="vertical"
                    aria-valuemax={100}
                    aria-valuemin={0}
                    aria-valuenow={Math.round(displayRatio(role) * 100)}
                    className="aim-role-guide-handle"
                    disabled={disabled}
                    type="button"
                    onKeyDown={(event) => handleKey(role, event)}
                    onPointerCancel={(event) => {
                      // Don't reset to `ratios` here: the user has already
                      // moved the handle to a new visual position. Snap-back
                      // on cancel was the source of a flicker that fought
                      // the parent's same-value short-circuit and the
                      // P0-B drag guard. If a server value is needed the
                      // outer useEffect will resync once the drag fully
                      // ends (draggingRef.current === null).
                      draggingRef.current = null;
                      event.currentTarget.releasePointerCapture(event.pointerId);
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
                  />
                  <span className="aim-role-guide-line after" aria-hidden="true" />
                </div>
              </div>
            );
          })}
        </div>
        <span className="aim-target-stage-caption">人物头皮为 0%，脚底为 100%；头部类型单独映射头皮到下巴。</span>
      </div>

      <div className="aim-role-controls">
        {ROLES.map((role) => {
          const meta = ROLE_META[role];
          return (
            <section className={`aim-role-control aim-role-control-${role}`} key={role}>
              <span className="aim-role-control-heading">
                <i />
                <b>{meta.label}</b>
                <small>{meta.caption}</small>
              </span>
              <ParameterNumberControl
                applyMode="save"
                disabled={disabled}
                kind="slider"
                label="框内垂直瞄点"
                max={100}
                min={0}
                onDraftChange={(next) => updateDraft(role, next / 100)}
                onCommit={(next) => commit(role, next / 100)}
                riskLevel="calibration"
                step={1}
                unit="%"
                value={Math.round(ratios[role] * 100)}
              />
            </section>
          );
        })}
      </div>
    </section>
  );
});
