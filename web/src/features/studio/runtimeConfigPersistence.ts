import {
  setRuntimeOutputGate,
  setRuntimeTriggerMode,
  updateRuntimeConfigField,
  type ConfigUpdateResponse,
  type RuntimeConfigValue
} from "../../api";

export function persistRuntimeConfigField(
  section: string,
  key: string,
  value: RuntimeConfigValue
): Promise<ConfigUpdateResponse> {
  if (section === "control" && key === "output_enabled") {
    if (typeof value !== "boolean") {
      throw new Error("物理输出开关需要布尔值。");
    }
    return setRuntimeOutputGate(value);
  }
  if (section === "control" && key === "trigger_mode") {
    if (typeof value !== "string") {
      throw new Error("触发模式需要字符串值。");
    }
    return setRuntimeTriggerMode(value);
  }
  return updateRuntimeConfigField(section, key, value);
}
