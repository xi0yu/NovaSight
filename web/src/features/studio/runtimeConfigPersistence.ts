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
  value: RuntimeConfigValue,
  expectedRevision?: number
): Promise<ConfigUpdateResponse> {
  if (section === "control" && key === "output_enabled") {
    if (typeof value !== "boolean") {
      throw new Error("物理输出开关需要布尔值。");
    }
    return setRuntimeOutputGate(value, expectedRevision);
  }
  if (section === "control" && key === "trigger_mode") {
    if (value !== "always" && value !== "hardware") {
      throw new Error("触发模式必须为 always 或 hardware。");
    }
    return setRuntimeTriggerMode(value, expectedRevision);
  }
  return updateRuntimeConfigField(section, key, value, expectedRevision);
}
