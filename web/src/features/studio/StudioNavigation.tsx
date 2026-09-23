import { useEffect, useRef } from "react";

import { NovaIcon, type NovaIconName } from "../../components/visual";

import "./studio-navigation.css";

export type ConsolePage =
  | "overview"
  | "onboarding"
  | "activity"
  | "device"
  | "capture"
  | "infer"
  | "control"
  | "models"
  | "management"
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

export type ConsolePageMetadata = {
  group: string;
  title: string;
  description: string;
};

export const DEFAULT_CONSOLE_PAGE: ConsolePage = "overview";

export const CONSOLE_PAGES = new Set<ConsolePage>([
  "overview",
  "onboarding",
  "activity",
  "device",
  "capture",
  "infer",
  "control",
  "models",
  "management",
  "license",
  "params",
  "control-test",
  "latency"
]);

const primaryNavigation: NavigationItem[] = [
  { id: "overview", label: "首页", detail: "当前状态与下一步", icon: "dashboard" },
  { id: "device", label: "设备", detail: "画面、运行与设置", icon: "devices" },
  { id: "models", label: "模型", detail: "模型资产与部署", icon: "models" },
  { id: "activity", label: "活动", detail: "最近发生了什么", icon: "notification" },
  { id: "management", label: "管理", detail: "对象、授权与开始使用", icon: "account" }
];

const deviceNavigation: NavigationItem[] = [
  { id: "device", label: "状态", detail: "当前核实状态", icon: "devices" },
  { id: "capture", label: "画面", detail: "输入与 ROI", icon: "capture" },
  { id: "infer", label: "推理", detail: "模型输入与识别结果", icon: "inference" },
  { id: "control", label: "目标与控制", detail: "选择、预测与输出", icon: "control" },
  { id: "params", label: "参数", detail: "调整、保存与生效", icon: "settings" },
  { id: "latency", label: "性能", detail: "链路时序与新鲜度", icon: "performance" },
  { id: "control-test", label: "控制测试", detail: "安全硬件实验", icon: "kmbox" }
];

const DEVICE_PAGES = new Set<ConsolePage>(deviceNavigation.map((item) => item.id));
const MANAGEMENT_PAGES = new Set<ConsolePage>(["management", "onboarding", "license"]);

export const CONSOLE_PAGE_METADATA: Record<ConsolePage, ConsolePageMetadata> = {
  overview: {
    group: "NovaSight",
    title: "首页",
    description: "查看当前设备、运行状态和现在需要注意的事情。"
  },
  onboarding: {
    group: "开始使用",
    title: "新手引导",
    description: "按任务顺序连接画面、选择模型、确认参数，再安全开始运行。"
  },
  activity: {
    group: "NovaSight",
    title: "活动",
    description: "查看最近发生的故障与操作结果；需要排查时再展开开发者详情。"
  },
  device: {
    group: "当前设备",
    title: "设备状态",
    description: "只展示服务当前已核实的连接、运行、识别和输出状态。"
  },
  capture: {
    group: "当前设备",
    title: "实时画面",
    description: "选择画面来源和清晰度，设置识别区域，并确认当前运行已经采用这些选择。"
  },
  infer: {
    group: "当前设备",
    title: "推理",
    description: "查看模型正在识别的画面，调整识别灵敏度，并确认结果保持新鲜。"
  },
  control: {
    group: "当前设备",
    title: "目标与控制",
    description: "观察选择主要目标、目标速度预测、连续非线性控制、输出限幅和命令输出。"
  },
  models: {
    group: "NovaSight",
    title: "模型",
    description: "从设备资产中选择候选模型，验证资格后部署，并核对真实装载状态。"
  },
  management: {
    group: "NovaSight",
    title: "管理",
    description: "在一个地方管理设备、模型、活动与授权。"
  },
  license: {
    group: "NovaSight",
    title: "授权",
    description: "查看当前授权范围，更换许可证，或在安全停止并确认输出后退出授权。"
  },
  params: {
    group: "当前设备",
    title: "参数",
    description: "按使用顺序设置何时响应、如何移动，以及是否把结果发送到设备。"
  },
  "control-test": {
    group: "当前设备",
    title: "控制测试",
    description: "脱离自动目标链路验证 kmNet 连接和受控移动输出。"
  },
  latency: {
    group: "当前设备",
    title: "性能",
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
  const activeItemRef = useRef<HTMLButtonElement>(null);
  const deviceActive = DEVICE_PAGES.has(activePage);

  useEffect(() => {
    if (typeof window.matchMedia !== "function" || !window.matchMedia("(max-width: 1024px)").matches) return;
    const item = activeItemRef.current;
    const scroller = item?.closest(".console-sidebar");
    if (!item || !(scroller instanceof HTMLElement)) return;
    const itemRect = item.getBoundingClientRect();
    const scrollerRect = scroller.getBoundingClientRect();
    scroller.scrollLeft += itemRect.left - scrollerRect.left - (scrollerRect.width - itemRect.width) / 2;
  }, [activePage]);

  return (
    <nav className="console-navigation" aria-label="NovaSight Studio 导航">
      <div className="console-primary-navigation">
        {primaryNavigation.map((item) => {
          const active = item.id === "device"
            ? deviceActive
            : item.id === "management"
              ? MANAGEMENT_PAGES.has(activePage)
              : activePage === item.id;
          return (
            <button type="button"
              className={active ? "console-nav active" : "console-nav"}
              key={item.id}
              onClick={() => onNavigate(item.id)}
              aria-label={item.label}
              aria-current={active ? "page" : undefined}
              ref={active ? activeItemRef : undefined}
              title={item.detail}
            >
              <span className="console-nav-icon" aria-hidden="true"><NovaIcon name={item.icon} size={18} /></span>
              <span className="console-nav-copy"><b>{item.label}</b><small>{item.detail}</small></span>
            </button>
          );
        })}
      </div>
      {deviceActive ? (
        <section className="console-device-navigation" aria-label="当前设备的深入功能">
          <span>当前设备</span>
          {deviceNavigation.map((item) => {
            const active = activePage === item.id;
            return (
              <button type="button"
                className={active ? "console-device-nav active" : "console-device-nav"}
                key={item.id}
                onClick={() => onNavigate(item.id)}
                aria-current={active ? "page" : undefined}
                ref={active ? activeItemRef : undefined}
                title={item.detail}
              >{item.label}</button>
            );
          })}
        </section>
      ) : null}
    </nav>
  );
}
