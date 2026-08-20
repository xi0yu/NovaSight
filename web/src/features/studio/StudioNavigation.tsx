import { NovaIcon, type NovaIconName } from "../../components/visual";

import "./studio-navigation.css";

export type ConsolePage =
  | "overview"
  | "capture"
  | "infer"
  | "control"
  | "models"
  | "license"
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

// M3b flips this only after the opt-in SHA passes the Jetson safe-start gate.
export const DEFAULT_CONSOLE_PAGE: ConsolePage = "capture";

export const CONSOLE_PAGES = new Set<ConsolePage>([
  "overview",
  "capture",
  "infer",
  "control",
  "models",
  "license",
  "params",
  "control-test",
  "latency"
]);

const navigationGroups: NavigationGroup[] = [
  {
    id: "runtime",
    label: "运行工作台",
    items: [
      { id: "overview", label: "运行总览", detail: "当前结论与恢复入口", icon: "dashboard" },
      { id: "capture", label: "采集", detail: "输入与 ROI", icon: "capture" },
      { id: "infer", label: "模型推理", detail: "模型与输出", icon: "inference" },
      { id: "control", label: "控制", detail: "目标与执行", icon: "control" }
    ]
  },
  {
    id: "configuration",
    label: "配置管理",
    items: [
      { id: "models", label: "模型管理", detail: "选择、验证与切换", icon: "models" },
      { id: "license", label: "授权管理", detail: "查看、更换与安全退出", icon: "shield-check" },
      { id: "params", label: "参数设置", detail: "触发到输出", icon: "settings" }
    ]
  },
  {
    id: "diagnostics",
    label: "排查工具",
    items: [
      { id: "control-test", label: "控制测试", detail: "硬件输出实验", icon: "kmbox" },
      { id: "latency", label: "延迟分析", detail: "采集链路时序", icon: "latency" }
    ]
  }
];

export const CONSOLE_PAGE_METADATA: Record<ConsolePage, ConsolePageMetadata> = {
  overview: {
    group: "运行工作台",
    title: "运行总览",
    description: "先确认生命周期、感知数据和硬件输出，再进入对应页面处理问题。"
  },
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
    description: "观察选择主要目标、目标速度预测、连续非线性控制、输出限幅和命令输出。"
  },
  models: {
    group: "配置管理",
    title: "模型管理",
    description: "浏览设备上的模型产物，经后端验证后切换，并以 active artifact 对账结果。"
  },
  license: {
    group: "配置管理",
    title: "授权管理",
    description: "查看当前授权范围，更换许可证，或在安全停止并确认输出后退出授权。"
  },
  params: {
    group: "配置管理",
    title: "参数设置",
    description: "按控制链顺序调整触发、延迟、算法、压枪、限幅和输出。"
  },
  "control-test": {
    group: "排查工具",
    title: "控制测试",
    description: "脱离自动目标链路验证 kmNet 连接和受控移动输出。"
  },
  latency: {
    group: "排查工具",
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
                <button type="button"
                  className={active ? "console-nav active" : "console-nav"}
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  aria-label={item.label}
                  aria-current={active ? "page" : undefined}
                  title={item.detail}
                >
                  <span className="console-nav-icon" aria-hidden="true">
                    <NovaIcon name={item.icon} size={18} />
                  </span>
                  <span className="console-nav-copy">
                    <b>{item.label}</b>
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
