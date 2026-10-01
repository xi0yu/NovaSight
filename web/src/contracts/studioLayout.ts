export const STUDIO_LAYOUT_PAGE_IDS = ["overview", "capture", "params", "fire", "activity", "settings", "about"] as const;

export type StudioLayoutPageId = typeof STUDIO_LAYOUT_PAGE_IDS[number];

export const STUDIO_LAYOUT_MODULE_IDS = {
  overview: ["setup", "runtime", "metrics", "pipeline", "diagnostics"],
  capture: ["source", "roi", "diagnostics"],
  params: ["response", "targeting", "motion", "output"],
  fire: ["stabilization"],
  activity: ["summary", "filters", "feed"],
  settings: ["backup", "restore", "details"],
  about: ["identity", "version", "capabilities", "support"],
} as const satisfies Record<StudioLayoutPageId, readonly string[]>;

const REQUIRED_MODULES: Record<StudioLayoutPageId, readonly string[]> = {
  overview: ["runtime"],
  capture: ["source"],
  params: ["response", "motion", "output", "targeting"],
  fire: ["stabilization"],
  activity: ["summary", "feed"],
  settings: ["backup", "restore"],
  about: ["identity"],
};

export type StudioLayoutPage = {
  id: StudioLayoutPageId;
  label: string;
  modules: string[];
};

export type StudioLayout = {
  schemaVersion: 1;
  revision: string;
  pages: StudioLayoutPage[];
};

export const DEFAULT_STUDIO_LAYOUT: StudioLayout = {
  schemaVersion: 1,
  revision: "builtin-2026-09",
  pages: [
    { id: "overview", label: "首页", modules: ["setup", "runtime", "metrics"] },
    { id: "capture", label: "设备管理", modules: [...STUDIO_LAYOUT_MODULE_IDS.capture] },
    { id: "params", label: "算法参数", modules: [...STUDIO_LAYOUT_MODULE_IDS.params] },
    { id: "fire", label: "开火稳定", modules: [...STUDIO_LAYOUT_MODULE_IDS.fire] },
    { id: "activity", label: "实时日志", modules: [...STUDIO_LAYOUT_MODULE_IDS.activity] },
    { id: "settings", label: "设置", modules: [...STUDIO_LAYOUT_MODULE_IDS.settings] },
    { id: "about", label: "关于", modules: [...STUDIO_LAYOUT_MODULE_IDS.about] },
  ],
};

export class StudioLayoutContractError extends Error {
  constructor(detail: string) {
    super(`Studio 页面配置无效：${detail}`);
    this.name = "StudioLayoutContractError";
  }
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new StudioLayoutContractError(`${path} 应为对象`);
  }
  return value as Record<string, unknown>;
}

function text(value: unknown, path: string, maxLength: number): string {
  if (typeof value !== "string" || value.trim() === "" || value.length > maxLength) {
    throw new StudioLayoutContractError(`${path} 应为 1-${maxLength} 个字符`);
  }
  return value.trim();
}

export function decodeStudioLayout(value: unknown): StudioLayout {
  const root = record(value, "layout");
  if (root.schemaVersion !== 1) {
    throw new StudioLayoutContractError("仅支持 schemaVersion=1");
  }
  if (!Array.isArray(root.pages)) {
    throw new StudioLayoutContractError("pages 应为数组");
  }

  const knownPages = new Set<string>(STUDIO_LAYOUT_PAGE_IDS);
  const seenPages = new Set<string>();
  const pages = root.pages.map((item, pageIndex): StudioLayoutPage => {
    const page = record(item, `pages[${pageIndex}]`);
    const id = text(page.id, `pages[${pageIndex}].id`, 24);
    if (!knownPages.has(id) || seenPages.has(id)) {
      throw new StudioLayoutContractError(`pages[${pageIndex}].id 不受支持或重复`);
    }
    seenPages.add(id);
    if (!Array.isArray(page.modules)) {
      throw new StudioLayoutContractError(`pages[${pageIndex}].modules 应为数组`);
    }
    const pageId = id as StudioLayoutPageId;
    const allowedModules = new Set<string>(STUDIO_LAYOUT_MODULE_IDS[pageId]);
    const seenModules = new Set<string>();
    const modules = page.modules.map((module, moduleIndex) => {
      const moduleId = text(module, `pages[${pageIndex}].modules[${moduleIndex}]`, 32);
      if (!allowedModules.has(moduleId) || seenModules.has(moduleId)) {
        throw new StudioLayoutContractError(`页面 ${pageId} 包含不受支持或重复的模块 ${moduleId}`);
      }
      seenModules.add(moduleId);
      return moduleId;
    });
    const missingModules = REQUIRED_MODULES[pageId].filter((moduleId) => !seenModules.has(moduleId));
    if (missingModules.length > 0) {
      throw new StudioLayoutContractError(`页面 ${pageId} 缺少必要模块 ${missingModules.join("、")}`);
    }
    return { id: pageId, label: text(page.label, `pages[${pageIndex}].label`, 12), modules };
  });

  const missingPages = STUDIO_LAYOUT_PAGE_IDS.filter((pageId) => !seenPages.has(pageId));
  if (missingPages.length > 0) {
    throw new StudioLayoutContractError(`缺少必要页面 ${missingPages.join("、")}`);
  }

  return {
    schemaVersion: 1,
    revision: text(root.revision, "revision", 64),
    pages,
  };
}

export function modulesForPage(layout: StudioLayout, pageId: StudioLayoutPageId): ReadonlySet<string> {
  return new Set(moduleOrderForPage(layout, pageId));
}

export function moduleOrderForPage(layout: StudioLayout, pageId: StudioLayoutPageId): readonly string[] {
  const page = layout.pages.find((candidate) => candidate.id === pageId);
  return page?.modules ?? STUDIO_LAYOUT_MODULE_IDS[pageId];
}
