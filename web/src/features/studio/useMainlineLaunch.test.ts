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
      "开启运行失败：当前授权不包含硬件控制。请使用包含硬件控制权限的正式许可证后，再从首页开启运行。"
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
