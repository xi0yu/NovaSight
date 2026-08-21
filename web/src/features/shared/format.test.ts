import { describe, expect, it } from "vitest";

import { formatRuntimeErrorMessage } from "./format";

describe("formatRuntimeErrorMessage", () => {
  it("turns the uncommissioned pointer error into an actionable Chinese message", () => {
    expect(formatRuntimeErrorMessage(
      "pointer device is not commissioned; configure hardware.auto_connect with a provisioned host and UUID",
    )).toBe("kmNet 尚未完成设备配置；请填写真实的地址、端口和 UUID 后保存。");
  });

  it("preserves unknown runtime messages for diagnostics", () => {
    expect(formatRuntimeErrorMessage("future runtime failure")).toBe("future runtime failure");
  });
});
