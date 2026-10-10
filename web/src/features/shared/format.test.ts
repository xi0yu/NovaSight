import { describe, expect, it } from "vitest";

import { ApiError } from "../../api";
import { formatRuntimeErrorMessage, getErrorMessage } from "./format";

describe("getErrorMessage", () => {
  it("explains catalog conflicts without hiding the original API error", () => {
    const error = new ApiError("model directory already exists: Arena", 409, { code: "MODEL_DIRECTORY_EXISTS" });
    expect(getErrorMessage(error)).toBe("这个文件夹已存在，请换一个名称。");
    expect(error.message).toBe("model directory already exists: Arena");
  });
});

describe("runtime failure guidance", () => {
  it.each([
    ["no perception adapter is installed for this daemon mode", "当前为主机预览模式，未启用 Jetson 感知适配器；可继续查看界面，真机运行请在 Jetson 上启动 NovaSight。"],
    ["pointer device is not commissioned; configure hardware.auto_connect with a provisioned host and UUID", "kmNet 尚未完成设备配置；请填写真实的地址、端口和 UUID 后保存。"],
    ["future runtime failure", "future runtime failure"],
  ])("explains %s while preserving unknown diagnostic messages", (message, expected) => {
    expect(formatRuntimeErrorMessage(message)).toBe(expected);
  });
});
