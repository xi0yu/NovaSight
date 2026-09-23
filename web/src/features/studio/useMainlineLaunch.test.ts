import { describe, expect, it } from "vitest";

import { ApiError } from "../../api";
import { formatMainlineLaunchError } from "./useMainlineLaunch";

describe("formatMainlineLaunchError", () => {
  it("explains how to run without licensed physical output", () => {
    const error = new ApiError(
      "license feature hardware_control is required",
      403,
      {
        code: "LICENSE_FEATURE_REQUIRED",
        required_feature: "hardware_control"
      }
    );

    expect(formatMainlineLaunchError(error)).toBe(
      "启动主链失败：当前授权不包含硬件控制。已保存配置开启了物理输出；如需先运行采集、推理和算法预览，请在“控制 → 输出”关闭“发送鼠标偏移”，或使用包含硬件控制权限的正式许可证。"
    );
  });

  it("does not describe unrelated feature failures as hardware control failures", () => {
    const error = new ApiError(
      "license feature runtime is required",
      403,
      {
        code: "LICENSE_FEATURE_REQUIRED",
        required_feature: "runtime"
      }
    );

    expect(formatMainlineLaunchError(error)).toContain(
      "license feature runtime is required"
    );
    expect(formatMainlineLaunchError(error)).not.toContain("开启了物理输出");
  });
});
