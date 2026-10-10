import type { CSSProperties } from "react";
import { NovaIcon } from "../../components/visual";

// Adapted from beUI Tabs (MIT, Saurabh Chauhan); see public/third-party-ui.txt.
// Four fixed sections use CSS for the sliding indicator, without Motion/Tailwind.
const SECTIONS = [
  { id: "response", label: "范围与触发", detail: "什么时候介入", icon: "target" },
  { id: "targeting", label: "目标与瞄点", detail: "跟随谁、瞄哪里", icon: "models" },
  { id: "motion", label: "移动与输出", detail: "调整跟随手感", icon: "prediction-line" },
  { id: "advanced", label: "进阶调校", detail: "标定、试算与诊断", icon: "settings" },
] as const;

export function AlgorithmTabs({ value, onChange }: {
  value: typeof SECTIONS[number]["id"];
  onChange: (value: typeof SECTIONS[number]["id"]) => void;
}) {
  const index = SECTIONS.findIndex((section) => section.id === value);
  return (
    <div className="algorithm-tabs" role="tablist" aria-label="算法调校分区" data-section={value}
      style={{ "--tab-index": index, "--tab-column": index % 2, "--tab-row": Math.floor(index / 2) } as CSSProperties}>
      <span className="algorithm-tab-indicator" aria-hidden="true" />
      {SECTIONS.map((item, itemIndex) => (
        <button key={item.id} id={`algorithm-tab-${item.id}`} data-section={item.id} type="button"
          role="tab" aria-label={item.label} aria-selected={value === item.id}
          aria-controls="algorithm-parameter-panel" tabIndex={value === item.id ? 0 : -1}
          onClick={() => onChange(item.id)}
          onKeyDown={(event) => {
            const next = event.key === "Home" ? 0 : event.key === "End" ? SECTIONS.length - 1
              : event.key === "ArrowRight" ? (itemIndex + 1) % SECTIONS.length
                : event.key === "ArrowLeft" ? (itemIndex + SECTIONS.length - 1) % SECTIONS.length : null;
            if (next === null) return;
            event.preventDefault();
            onChange(SECTIONS[next].id);
            event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus();
          }}>
          <span className="algorithm-tab-icon"><NovaIcon name={item.icon} size={18} /></span>
          <span>{item.label}<small>{item.detail}</small></span>
        </button>
      ))}
    </div>
  );
}
