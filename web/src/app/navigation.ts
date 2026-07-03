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
  feature?: string;
};

export const studioNavItems: StudioNavItem[] = [
  { id: "dashboard", label: "总览", hint: "运行状态与链路健康" },
  { id: "devices", label: "设置", hint: "采集、推理与算法参数" },
  { id: "models", label: "模型", hint: "模型选择与场景使用" },
  { id: "config", label: "配置", hint: "运行参数与系统设置" },
  { id: "license", label: "授权", hint: "卡密与本机许可状态" }
];
