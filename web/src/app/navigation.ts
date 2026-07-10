import type { NovaIconName } from "../components/visual";

export type StudioViewId =
  | "dashboard"
  | "devices"
  | "models"
  | "config"
  | "license";

export type StudioNavItem = {
  id: StudioViewId;
  label: string;
  hint: string;
  icon: NovaIconName;
  feature?: string;
};

export const studioNavItems: StudioNavItem[] = [
  { id: "dashboard", label: "总览", hint: "运行状态与链路健康", icon: "dashboard" },
  { id: "devices", label: "设置", hint: "采集、推理与算法参数", icon: "capture" },
  { id: "models", label: "模型", hint: "模型选择与场景使用", icon: "models" },
  { id: "config", label: "配置", hint: "运行参数与系统设置", icon: "settings" },
  { id: "license", label: "授权", hint: "卡密与本机许可状态", icon: "shield-check" }
];
