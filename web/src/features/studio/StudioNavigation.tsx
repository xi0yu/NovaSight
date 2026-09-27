import { useEffect, useRef } from "react";

import { NovaIcon, type NovaIconName } from "../../components/visual";
import { DEFAULT_STUDIO_LAYOUT, type StudioLayoutPage } from "../../contracts/studioLayout";

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
  | "latency"
  | "settings"
  | "about";

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
  "overview", "onboarding", "activity", "device", "capture", "infer", "control",
  "models", "management", "license", "params", "control-test", "latency", "about",
  "settings",
]);

const navigationDetails: Record<StudioLayoutPage["id"], Omit<NavigationItem, "id" | "label">> = {
  overview: { detail: "状态、实时数据与下一步", icon: "dashboard" },
  capture: { detail: "输入、模型与采集规格", icon: "devices" },
  params: { detail: "识别、目标与输出参数", icon: "settings" },
  activity: { detail: "按等级查看运行与操作日志", icon: "logs" },
  settings: { detail: "备份、恢复与系统设置", icon: "settings" },
  about: { detail: "平台版本、能力与支持", icon: "help" },
};

const CAPTURE_PAGES = new Set<ConsolePage>(["device", "capture", "infer", "control", "models", "control-test", "latency"]);

function selectedPrimaryPage(page: ConsolePage): StudioLayoutPage["id"] {
  if (CAPTURE_PAGES.has(page)) return "capture";
  if (page === "management" || page === "license") return "about";
  if (page === "onboarding") return "overview";
  return page as StudioLayoutPage["id"];
}

export const CONSOLE_PAGE_METADATA: Record<ConsolePage, ConsolePageMetadata> = {
  overview: { group: "NovaSight", title: "首页", description: "先看设备是否正常，再看实时数据和下一步操作。" },
  onboarding: { group: "开始使用", title: "新手引导", description: "按任务顺序连接画面、选择模型、确认参数，再安全开始运行。" },
  activity: { group: "NovaSight", title: "实时日志", description: "按严重程度查看运行与操作记录；需要排查时再展开开发者详情。" },
  device: { group: "设备管理", title: "设备状态", description: "只展示服务当前已核实的连接、运行、识别和输出状态。" },
  capture: { group: "NovaSight", title: "设备管理", description: "选择送入模型的画面来源，并确认设备实际采用的规格。" },
  infer: { group: "设备管理", title: "推理与模型", description: "查看模型正在识别的画面，调整识别灵敏度，并确认结果保持新鲜。" },
  control: { group: "算法参数", title: "目标与控制", description: "观察目标选择、速度预测、连续控制、输出限幅和命令输出。" },
  models: { group: "设备管理", title: "模型", description: "从设备资产中选择模型，验证资格后部署，并核对真实装载状态。" },
  management: { group: "关于 NovaSight", title: "资源与授权", description: "集中查看设备、模型、活动和授权。" },
  license: { group: "关于 NovaSight", title: "授权", description: "查看当前授权范围，或在安全停止并确认输出后更换授权。" },
  params: { group: "NovaSight", title: "算法参数", description: "调整目标锁定、移动手感与安全边界。" },
  "control-test": { group: "设备管理", title: "控制测试", description: "脱离自动目标链路验证 kmNet 连接和受控移动输出。" },
  latency: { group: "设备管理", title: "性能", description: "定位采集到设备发送之间的阶段耗时和数据新鲜度问题。" },
  settings: { group: "NovaSight", title: "设置", description: "备份当前设置，或从已有备份安全恢复。" },
  about: { group: "NovaSight", title: "关于 NovaSight", description: "查看平台版本、能力边界、授权状态和支持入口。" },
};

export function StudioNavigation({
  activePage,
  onNavigate,
  pages = DEFAULT_STUDIO_LAYOUT.pages,
}: {
  activePage: ConsolePage;
  onNavigate: (page: ConsolePage) => void;
  pages?: StudioLayoutPage[];
}) {
  const activeItemRef = useRef<HTMLButtonElement>(null);
  const selectedPage = selectedPrimaryPage(activePage);

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
        {pages.map((configuredPage) => {
          const item: NavigationItem = {
            id: configuredPage.id,
            label: configuredPage.label,
            ...navigationDetails[configuredPage.id],
          };
          const active = selectedPage === item.id;
          return (
            <button
              aria-current={active ? "page" : undefined}
              aria-label={item.label}
              className={active ? "console-nav active" : "console-nav"}
              key={item.id}
              onClick={() => onNavigate(item.id)}
              ref={active ? activeItemRef : undefined}
              title={item.detail}
              type="button"
            >
              <span className="console-nav-icon" aria-hidden="true"><NovaIcon name={item.icon} size={19} /></span>
              <span className="console-nav-copy"><b>{item.label}</b><small>{item.detail}</small></span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
