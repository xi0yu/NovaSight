import { NovaIcon, type NovaIconName } from "../../components/visual";

import "./studio-navigation.css";

export type ConsolePage =
  | "capture"
  | "infer"
  | "control"
  | "params"
  | "control-test"
  | "latency";

type NavigationItem = {
  id: ConsolePage;
  label: string;
  detail: string;
  icon: NovaIconName;
};

type NavigationGroup = {
  id: "runtime" | "configuration" | "diagnostics";
  label: string;
  items: NavigationItem[];
};

export type ConsolePageMetadata = {
  group: string;
  title: string;
  description: string;
};

export const DEFAULT_CONSOLE_PAGE: ConsolePage = "capture";

export const CONSOLE_PAGES = new Set<ConsolePage>([
  "capture",
  "infer",
  "control",
  "params",
  "control-test",
  "latency"
]);

const navigationGroups: NavigationGroup[] = [
  {
    id: "runtime",
    label: "运行工作台",
    items: [
      { id: "capture", label: "采集", detail: "输入与 ROI", icon: "capture" },
      { id: "infer", label: "模型推理", detail: "模型与输出", icon: "inference" },
      { id: "control", label: "控制", detail: "目标与执行", icon: "control" }
    ]
  },
  {
    id: "configuration",
    label: "配置管理",
    items: [
      { id: "params", label: "参数设置", detail: "类别、算法与跟踪", icon: "settings" }
    ]
  },
  {
    id: "diagnostics",
    label: "诊断工具",
    items: [
      { id: "control-test", label: "控制测试", detail: "硬件输出实验", icon: "kmbox" },
      { id: "latency", label: "延迟分析", detail: "采集链路时序", icon: "latency" }
    ]
  }
];

export const CONSOLE_PAGE_METADATA: Record<ConsolePage, ConsolePageMetadata> = {
  capture: {
    group: "运行工作台",
    title: "采集",
    description: "选择输入设备和采集规格，确认 ROI 与最新帧链路处于可用状态。"
  },
  infer: {
    group: "运行工作台",
    title: "模型推理",
    description: "管理当前模型、推理预览和识别结果。"
  },
  control: {
    group: "运行工作台",
    title: "控制",
    description: "观察目标选择、Atan 控制量和设备发送是否形成稳定闭环。"
  },
  params: {
    group: "配置管理",
    title: "参数设置",
    description: "查看当前配置，维护类别、控制算法和设备参数。"
  },
  "control-test": {
    group: "诊断工具",
    title: "控制测试",
    description: "脱离自动目标链路验证 kmNet 连接和受控移动输出。"
  },
  latency: {
    group: "诊断工具",
    title: "延迟分析",
    description: "定位采集到设备发送之间的阶段耗时和数据新鲜度问题。"
  }
};

export function StudioNavigation({
  activePage,
  onNavigate
}: {
  activePage: ConsolePage;
  onNavigate: (page: ConsolePage) => void;
}) {
  return (
    <nav className="console-navigation" aria-label="NovaSight Studio 导航">
      {navigationGroups.map((group) => (
        <section className="console-nav-group" key={group.id} aria-labelledby={`console-nav-${group.id}`}>
          <h2 id={`console-nav-${group.id}`}>{group.label}</h2>
          <div className="console-nav-group-items">
            {group.items.map((item) => {
              const active = activePage === item.id;
              return (
                <button
                  className={active ? "console-nav active" : "console-nav"}
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  type="button"
                  aria-current={active ? "page" : undefined}
                >
                  <span className="console-nav-icon" aria-hidden="true">
                    <NovaIcon name={item.icon} size={18} />
                  </span>
                  <span className="console-nav-copy">
                    <b>{item.label}</b>
                    <small>{item.detail}</small>
                  </span>
                </button>
              );
            })}
          </div>
        </section>
      ))}
    </nav>
  );
}
