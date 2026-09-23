import { useEffect, useRef } from "react";

import { NovaIcon, type NovaIconName } from "../../components/visual";

import "./studio-navigation.css";

export type ConsolePage =
  | "overview"
  | "activity"
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

export type ConsolePageMetadata = {
  group: string;
  title: string;
  description: string;
};

export const DEFAULT_CONSOLE_PAGE: ConsolePage = "overview";

export const CONSOLE_PAGES = new Set<ConsolePage>([
  "overview",
  "activity",
  "capture",
  "infer",
  "control",
  "models",
  "license",
  "params",
  "control-test",
  "latency"
]);

const primaryNavigation: NavigationItem[] = [
  { id: "overview", label: "首页", detail: "当前状态与下一步", icon: "dashboard" },
  { id: "capture", label: "设备", detail: "画面、运行与设置", icon: "devices" },
  { id: "models", label: "模型", detail: "模型资产与部署", icon: "models" },
  { id: "activity", label: "活动", detail: "最近发生了什么", icon: "notification" },
  { id: "license", label: "授权", detail: "权限与安全退出", icon: "account" }
];

const deviceNavigation: NavigationItem[] = [
  { id: "capture", label: "画面", detail: "输入与 ROI", icon: "capture" },
  { id: "infer", label: "推理", detail: "模型输入与识别结果", icon: "inference" },
  { id: "control", label: "目标与控制", detail: "选择、预测与输出", icon: "control" },
  { id: "params", label: "参数", detail: "调整、保存与生效", icon: "settings" },
  { id: "latency", label: "性能", detail: "链路时序与新鲜度", icon: "performance" },
  { id: "control-test", label: "控制测试", detail: "安全硬件实验", icon: "kmbox" }
];

const DEVICE_PAGES = new Set<ConsolePage>(deviceNavigation.map((item) => item.id));

export const CONSOLE_PAGE_METADATA: Record<ConsolePage, ConsolePageMetadata> = {
  overview: {
    group: "NovaSight",
    title: "首页",
    description: "查看当前设备、运行状态和现在需要注意的事情。"
  },
  activity: {
    group: "NovaSight",
    title: "活动",
    description: "查看最近发生的故障与操作结果；需要排查时再展开开发者详情。"
  },
  capture: {
    group: "当前设备",
    title: "实时画面",
    description: "选择输入设备和采集规格，确认 ROI 与最新帧链路处于可用状态。"
  },
  infer: {
    group: "当前设备",
    title: "推理",
    description: "管理当前模型、推理预览和识别结果。"
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
  license: {
    group: "NovaSight",
    title: "授权",
    description: "查看当前授权范围，更换许可证，或在安全停止并确认输出后退出授权。"
  },
  params: {
    group: "当前设备",
    title: "参数",
    description: "沿控制因果链调整参数，并区分草稿、已保存配置与运行态生效结果。"
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
          const active = item.id === "capture" ? deviceActive : activePage === item.id;
          return (
            <button type="button"
              className={active ? "console-nav active" : "console-nav"}
              key={item.id}
              onClick={() => onNavigate(item.id)}
              aria-label={item.label}
              aria-current={active ? "page" : undefined}
              ref={activePage === item.id ? activeItemRef : undefined}
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
