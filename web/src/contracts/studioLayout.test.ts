import { describe, expect, it } from "vitest";

import { DEFAULT_STUDIO_LAYOUT, StudioLayoutContractError, decodeStudioLayout } from "./studioLayout";

describe("studio layout contract", () => {
  it("accepts reordering known pages and modules", () => {
    const layout = decodeStudioLayout({
      schemaVersion: 1,
      revision: "ops-42",
      pages: [
        { id: "activity", label: "实时日志", modules: ["feed", "summary", "filters"] },
        { id: "overview", label: "首页", modules: ["metrics", "runtime"] },
        { id: "capture", label: "设备管理", modules: ["source", "model"] },
        { id: "params", label: "算法参数", modules: ["output", "targeting", "motion", "response"] },
        { id: "about", label: "关于", modules: ["identity", "version"] },
      ],
    });

    expect(layout.revision).toBe("ops-42");
    expect(layout.pages[0]?.modules).toEqual(["feed", "summary", "filters"]);
  });

  it("rejects arbitrary pages and modules instead of executing remote UI", () => {
    expect(() => decodeStudioLayout({
      schemaVersion: 1,
      revision: "unsafe",
      pages: [{ id: "overview", label: "首页", modules: ["remote-html"] }],
    })).toThrow(StudioLayoutContractError);
  });

  it("keeps a built-in complete fallback", () => {
    expect(DEFAULT_STUDIO_LAYOUT.pages.map((page) => page.id)).toEqual([
      "overview", "capture", "params", "activity", "about",
    ]);
  });

  it("rejects a layout that hides required tasks or the output control", () => {
    expect(() => decodeStudioLayout({
      ...DEFAULT_STUDIO_LAYOUT,
      pages: DEFAULT_STUDIO_LAYOUT.pages.filter((page) => page.id !== "activity"),
    })).toThrow(/缺少必要页面 activity/);
    expect(() => decodeStudioLayout({
      ...DEFAULT_STUDIO_LAYOUT,
      pages: DEFAULT_STUDIO_LAYOUT.pages.map((page) => page.id === "params"
        ? { ...page, modules: page.modules.filter((module) => module !== "output") }
        : page),
    })).toThrow(/缺少必要模块 output/);
  });
});
