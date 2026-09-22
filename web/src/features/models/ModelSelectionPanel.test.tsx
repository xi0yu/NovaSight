import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelCatalogDirectory, ModelCatalogModel } from "../../api";

import { ModelSelectionPanel } from "./ModelSelectionPanel";

describe("ModelSelectionPanel", () => {
  beforeEach(() => window.sessionStorage.clear());

  const stableModel: ModelCatalogModel = {
    type: "model", name: "stable.engine", relative_path: "stable.engine", kind: "engine",
    size_bytes: 1024, scan_status: "ready", scan_reason: "", recommendation: "recommended", tags: ["稳定"],
  };
  const otherModel: ModelCatalogModel = {
    ...stableModel, name: "other.engine", relative_path: "other.engine", recommendation: "unrated", tags: [],
  };
  const catalog: ModelCatalogDirectory = {
    type: "directory", name: "models", relative_path: "", children: [stableModel, otherModel],
  };
  function panelProps(overrides: Record<string, unknown> = {}) {
    return {
      root: catalog, loading: false, directoryCount: 1, modelCount: 2,
      selectedPath: stableModel.relative_path, selectedModel: stableModel,
      selectedArtifact: null, selectedVersion: null, activeArtifactId: null, activeArtifactPath: "",
      runtimeBackend: "", runtimeInputShape: "", catalogMessage: "", switchMessage: "",
      busy: null, canSwitch: true, parserPreset: "auto" as const,
      onParserPresetChange: vi.fn(), onRefresh: vi.fn(), onCreateFolder: vi.fn(async () => {}), onRequestMove: vi.fn(), onSelectModel: vi.fn(),
      onSaveMetadata: vi.fn(), onSwitch: vi.fn(), ...overrides,
    };
  }

  it("keeps saved tag filters while the catalog is still loading", () => {
    window.sessionStorage.setItem("novasight.model-filters.v1", JSON.stringify({ recommendation: "all", tags: ["稳定"] }));
    const view = render(<ModelSelectionPanel {...panelProps({ root: null, loading: true })} />);
    view.rerender(<ModelSelectionPanel {...panelProps()} />);
    expect(screen.getByRole("button", { name: "稳定" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText(/1 个模型 · 清除筛选后返回当前文件夹/)).toBeInTheDocument();
  });

  it("explains which files are registered, need validation, or are view-only", () => {
    const registered = { ...stableModel, artifact_id: 12 };
    const viewOnly = { ...otherModel, kind: "onnx" as const, name: "preview.onnx", relative_path: "preview.onnx" };
    render(<ModelSelectionPanel {...panelProps({
      root: { ...catalog, children: [registered, otherModel, viewOnly] },
      modelCount: 3,
      selectedModel: registered,
    })} />);
    expect(screen.getByText(/1 个已登记 Engine · 1 个待验证 Engine · 1 个仅供查看/)).toBeVisible();
  });

  it("filters by actual file usage without changing deployment or losing the selected model", async () => {
    const registered = { ...stableModel, artifact_id: 12 };
    const viewOnly = { ...otherModel, kind: "onnx" as const, name: "preview.onnx", relative_path: "preview.onnx" };
    const onSwitch = vi.fn();
    render(<ModelSelectionPanel {...panelProps({
      root: { ...catalog, children: [registered, otherModel, viewOnly] },
      modelCount: 3,
      selectedModel: registered,
      activeArtifactId: 12,
      onSwitch,
    })} />);
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "文件使用状态" }), "unregistered_engine");
    expect(screen.getByRole("button", { name: /other.engine，路径 other.engine/ })).toBeVisible();
    expect(screen.queryByRole("button", { name: /stable.engine，路径 stable.engine/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /preview.onnx，路径 preview.onnx/ })).not.toBeInTheDocument();
    expect(screen.getByText(/当前筛选隐藏了已选模型/)).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "清除筛选" }));
    expect(screen.getByRole("button", { name: /stable.engine，路径 stable.engine/ })).toBeVisible();
    expect(onSwitch).not.toHaveBeenCalled();
  });

  it("returns to the first page after the operator changes file usage", async () => {
    const files: ModelCatalogModel[] = Array.from({ length: 73 }, (_, index) => ({
      ...otherModel,
      name: `engine-${index}.engine`,
      relative_path: `engine-${index}.engine`,
      artifact_id: index === 0 ? 12 : undefined,
    }));
    render(<ModelSelectionPanel {...panelProps({
      root: { ...catalog, children: files },
      modelCount: files.length,
      selectedModel: null,
      selectedPath: undefined,
    })} />);
    await userEvent.click(screen.getByRole("button", { name: "再显示 24 个模型" }));
    expect(screen.getAllByRole("button", { name: /engine-\d+\.engine，路径/ })).toHaveLength(48);
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "文件使用状态" }), "registered");
    expect(screen.getAllByRole("button", { name: /engine-\d+\.engine，路径/ })).toHaveLength(1);
    await userEvent.click(screen.getByRole("button", { name: "清除筛选" }));
    expect(screen.getAllByRole("button", { name: /engine-\d+\.engine，路径/ })).toHaveLength(24);
  });

  it("does not switch a model hidden by the current filters", async () => {
    const onSwitch = vi.fn();
    render(<ModelSelectionPanel {...panelProps({ onSwitch })} />);
    await userEvent.click(within(screen.getByRole("group", { name: "推荐状态筛选" })).getByRole("button", { name: "待整理" }));
    expect(screen.getByText(/当前筛选隐藏了已选模型/)).toBeVisible();
    expect(screen.getByRole("button", { name: "验证并切换到所选模型" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /other.engine，路径 other.engine/ })).toBeVisible();
    expect(onSwitch).not.toHaveBeenCalled();
  });

  it("finds files by path and keeps the switch action beside the selection", async () => {
    render(<ModelSelectionPanel {...panelProps()} />);
    await userEvent.type(screen.getByRole("searchbox", { name: "查找模型文件" }), "other.engine");
    expect(screen.getByRole("button", { name: /other.engine，路径 other.engine/ })).toBeVisible();
    expect(screen.queryByRole("button", { name: /stable.engine，路径 stable.engine/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "验证并切换到所选模型" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "清除筛选" }));
    expect(screen.getByRole("button", { name: /stable.engine，路径 stable.engine/ })).toBeVisible();
    expect(screen.getByRole("button", { name: "验证并切换到所选模型" })).toBeEnabled();
  });

  it("includes a typed tag when saving and protects an unsaved metadata draft", async () => {
    const onSaveMetadata = vi.fn();
    render(<ModelSelectionPanel {...panelProps({ onSaveMetadata })} />);
    expect(screen.getByRole("region", { name: "整理此模型" })).toBeVisible();
    await userEvent.type(screen.getByRole("textbox", { name: "新增模型标签" }), "低延迟");
    expect(screen.getByText(/请先保存或放弃/)).toBeVisible();
    expect(screen.getByRole("button", { name: /other.engine/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: "全部文件夹" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "刷新模型" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "保存整理结果" }));
    expect(onSaveMetadata).toHaveBeenCalledWith("recommended", ["稳定", "低延迟"]);
  });

  it("lets the operator explicitly discard metadata edits before choosing another model", async () => {
    render(<ModelSelectionPanel {...panelProps()} />);
    await userEvent.click(within(screen.getByRole("group", { name: "模型推荐状态" })).getByRole("button", { name: "不推荐" }));
    expect(screen.getByRole("button", { name: /other.engine/ })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "放弃整理修改" }));
    expect(screen.getByRole("button", { name: /other.engine/ })).toBeEnabled();
  });

  it("does not offer editable metadata for ONNX files that cannot be saved", async () => {
    const onnx = { ...otherModel, kind: "onnx" as const, name: "other.onnx", relative_path: "other.onnx" };
    render(<ModelSelectionPanel {...panelProps({ root: { ...catalog, children: [onnx] }, selectedModel: onnx, selectedPath: onnx.relative_path })} />);
    expect(screen.getByRole("textbox", { name: "新增模型标签" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "保存整理结果" })).toBeDisabled();
  });

  it("shows a direct recovery path instead of empty filters when no model exists", () => {
    render(
      <ModelSelectionPanel
        root={null}
        loading={false}
        directoryCount={0}
        modelCount={0}
        selectedPath={undefined}
        selectedModel={null}
        selectedArtifact={null}
        selectedVersion={null}
        activeArtifactId={null}
        activeArtifactPath=""
        runtimeBackend=""
        runtimeInputShape=""
        catalogMessage=""
        switchMessage=""
        busy={null}
        canSwitch={false}
        parserPreset="auto"
        onParserPresetChange={vi.fn()}
        onRefresh={vi.fn()}
        onCreateFolder={vi.fn(async () => {})}
        onRequestMove={vi.fn()}
        onSelectModel={vi.fn()}
        onSaveMetadata={vi.fn()}
        onSwitch={vi.fn()}
      />
    );

    expect(screen.queryByRole("region", { name: "模型筛选" })).not.toBeInTheDocument();
    expect(screen.getByText(/将 \.engine 文件放入设备的 models 目录/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "刷新模型" })).toBeEnabled();
    expect(screen.queryByText("所选 Engine 文件")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "验证并切换到所选模型" })).not.toBeInTheDocument();
  });

  it("restores model filters when the operator returns during the same browser session", async () => {
    const root: ModelCatalogDirectory = {
      type: "directory",
      name: "models",
      relative_path: "",
      children: [
        {
          type: "model",
          name: "stable.engine",
          relative_path: "stable.engine",
          kind: "engine",
          size_bytes: 1024,
          scan_status: "ready",
          scan_reason: "",
          recommendation: "recommended",
          tags: ["稳定"],
        },
      ],
    };
    const props = {
      root,
      loading: false,
      directoryCount: 1,
      modelCount: 1,
      selectedPath: undefined,
      selectedModel: null,
      selectedArtifact: null,
      selectedVersion: null,
      activeArtifactId: null,
      activeArtifactPath: "",
      runtimeBackend: "",
      runtimeInputShape: "",
      catalogMessage: "",
      switchMessage: "",
      busy: null,
      canSwitch: false,
      parserPreset: "auto" as const,
      onParserPresetChange: vi.fn(),
      onRefresh: vi.fn(),
      onCreateFolder: vi.fn(async () => {}),
      onRequestMove: vi.fn(),
      onSelectModel: vi.fn(),
      onSaveMetadata: vi.fn(),
      onSwitch: vi.fn(),
    };

    const firstRender = render(<ModelSelectionPanel {...props} />);
    await userEvent.click(within(screen.getByRole("group", { name: "推荐状态筛选" })).getByRole("button", { name: "推荐" }));
    await userEvent.click(screen.getByRole("button", { name: "稳定" }));
    firstRender.unmount();

    render(<ModelSelectionPanel {...props} />);
    expect(within(screen.getByRole("group", { name: "推荐状态筛选" })).getByRole("button", { name: "推荐" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "稳定" })).toHaveAttribute("aria-pressed", "true");
  });

  it("keeps advanced model controls behind a disclosure without applying the selected file", async () => {
    const model = {
      type: "model" as const,
      name: "stable.engine",
      relative_path: "stable.engine",
      kind: "engine" as const,
      size_bytes: 1024,
      scan_status: "ready" as const,
      scan_reason: "",
      recommendation: "recommended" as const,
      tags: ["稳定"],
    };
    const onSwitch = vi.fn();
    render(<ModelSelectionPanel
      root={{ type: "directory", name: "models", relative_path: "", children: [model] }}
      loading={false} directoryCount={1} modelCount={1}
      selectedPath="stable.engine" selectedModel={model} selectedArtifact={null} selectedVersion={null}
      activeArtifactId={null} activeArtifactPath="" runtimeBackend="" runtimeInputShape=""
      catalogMessage="" switchMessage="" busy={null} canSwitch
      parserPreset="auto" onParserPresetChange={vi.fn()} onRefresh={vi.fn()} onCreateFolder={vi.fn(async () => {})} onRequestMove={vi.fn()}
      onSelectModel={vi.fn()} onSaveMetadata={vi.fn()} onSwitch={onSwitch}
    />);

    expect(screen.getByText(/此文件尚未部署/)).toBeVisible();
    expect(screen.getByRole("combobox", { name: "模型解析格式" })).not.toBeVisible();
    await userEvent.click(screen.getByText("查看文件详情与解析设置"));
    expect(screen.getByRole("combobox", { name: "模型解析格式" })).toBeVisible();
    expect(onSwitch).not.toHaveBeenCalled();
  });

  it("does not mistake an unregistered file for the active deployment", () => {
    render(<ModelSelectionPanel {...panelProps({ selectedArtifact: null, activeArtifactId: null })} />);
    expect(screen.getByText(/此文件尚未部署/)).toBeVisible();
    expect(screen.getByRole("button", { name: "验证并切换到所选模型" })).toBeEnabled();
  });

  it("shows the actual deployed state and does not offer a no-op switch", () => {
    const registered = { ...stableModel, artifact_id: 12, project_name: "Kenny", artifact_status: "ready" };
    render(<ModelSelectionPanel {...panelProps({
      root: { ...catalog, children: [registered] }, selectedModel: registered,
      activeArtifactId: 12, activeArtifactPath: "/models/stable.engine", activeLoaded: false,
    })} />);
    expect(screen.getByText(/此文件已部署，但当前尚未装载/)).toBeVisible();
    expect(screen.getByRole("button", { name: "已是当前部署模型" })).toBeDisabled();
    expect(screen.getByText("Kenny")).toBeVisible();
  });

  it("finds a model by its project or tag", async () => {
    const registered = { ...stableModel, project_name: "Kenny" };
    render(<ModelSelectionPanel {...panelProps({ root: { ...catalog, children: [registered, otherModel] }, selectedModel: registered })} />);
    await userEvent.type(screen.getByRole("searchbox", { name: "查找模型文件" }), "Kenny");
    expect(screen.getByRole("button", { name: /stable.engine，路径 stable.engine/ })).toBeVisible();
    expect(screen.queryByRole("button", { name: /other.engine，路径 other.engine/ })).not.toBeInTheDocument();
  });

  it("browses nested folders and searches across the entire catalog", async () => {
    const nested = { ...otherModel, name: "arena.engine", relative_path: "Arena/v2/arena.engine" };
    const root: ModelCatalogDirectory = { ...catalog, children: [stableModel, {
      type: "directory", name: "Arena", relative_path: "Arena", children: [{
        type: "directory", name: "v2", relative_path: "Arena/v2", children: [nested],
      }],
    }] };
    render(<ModelSelectionPanel {...panelProps({ root, modelCount: 2, directoryCount: 2 })} />);

    await userEvent.click(screen.getByRole("button", { name: "打开文件夹 Arena，包含 1 个模型" }));
    expect(screen.queryByRole("button", { name: /stable.engine，路径 stable.engine/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "打开文件夹 v2，包含 1 个模型" }));
    expect(screen.getByRole("button", { name: /arena.engine，路径 Arena\/v2\/arena.engine/ })).toBeVisible();
    expect(screen.getByRole("button", { name: "v2" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByText("位置：stable.engine")).toBeVisible();

    await userEvent.type(screen.getByRole("searchbox", { name: "查找模型文件" }), "stable.engine");
    expect(screen.getByText("全部文件夹的筛选结果")).toBeVisible();
    expect(screen.getByRole("button", { name: /stable.engine，路径 stable.engine/ })).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "清除筛选" }));
    expect(screen.getByRole("button", { name: /arena.engine，路径 Arena\/v2\/arena.engine/ })).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "全部文件夹" }));
    expect(screen.getByRole("button", { name: "打开文件夹 Arena，包含 1 个模型" })).toBeVisible();
  });

  it("creates a folder from an empty model catalog without changing the deployment", async () => {
    const onCreateFolder = vi.fn(async () => {});
    render(<ModelSelectionPanel {...panelProps({
      root: { ...catalog, children: [] }, modelCount: 0, directoryCount: 0,
      selectedModel: null, selectedPath: undefined, canSwitch: false, onCreateFolder,
    })} />);
    await userEvent.click(screen.getByRole("button", { name: "新建文件夹" }));
    await userEvent.type(screen.getByLabelText("在 全部文件夹 中新建文件夹"), "Arena");
    await userEvent.click(screen.getByRole("button", { name: "创建" }));
    expect(onCreateFolder).toHaveBeenCalledWith("Arena");
    await waitFor(() => expect(screen.queryByLabelText("在 全部文件夹 中新建文件夹")).not.toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "验证并切换到所选模型" })).not.toBeInTheDocument();
  });

  it("keeps the folder name and backend reason when creation fails", async () => {
    const onCreateFolder = vi.fn(async () => { throw new Error("文件夹已存在"); });
    render(<ModelSelectionPanel {...panelProps({ onCreateFolder })} />);
    await userEvent.click(screen.getByRole("button", { name: "新建文件夹" }));
    await userEvent.type(screen.getByLabelText("在 全部文件夹 中新建文件夹"), "Arena");
    await userEvent.click(screen.getByRole("button", { name: "创建" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("文件夹已存在");
    expect(screen.getByLabelText("在 全部文件夹 中新建文件夹")).toHaveValue("Arena");
  });

  it("requests an explicit file move without switching the deployed model", async () => {
    const onRequestMove = vi.fn();
    const onSwitch = vi.fn();
    const root: ModelCatalogDirectory = { ...catalog, children: [stableModel, {
      type: "directory", name: "Arena", relative_path: "Arena", children: []
    }] };
    render(<ModelSelectionPanel {...panelProps({ root, onRequestMove, onSwitch })} />);
    await userEvent.click(screen.getByRole("button", { name: "移动或改名文件" }));
    await userEvent.clear(screen.getByRole("textbox", { name: "新文件名（.engine）" }));
    await userEvent.type(screen.getByRole("textbox", { name: "新文件名（.engine）" }), "renamed");
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "移到文件夹" }), "Arena");
    await userEvent.click(screen.getByRole("button", { name: "检查并确认" }));
    expect(onRequestMove).toHaveBeenCalledWith("stable.engine", "Arena/renamed.engine");
    expect(onSwitch).not.toHaveBeenCalled();
  });

  it("does not offer direct file moves for registered or active models", () => {
    const registered = { ...stableModel, artifact_id: 7 };
    render(<ModelSelectionPanel {...panelProps({ root: { ...catalog, children: [registered] }, selectedModel: registered, activeArtifactId: 7 })} />);
    expect(screen.getByRole("button", { name: "移动或改名文件" })).toBeDisabled();
  });

  it("remembers the operator's sorting choice for this browser session", async () => {
    const first = render(<ModelSelectionPanel {...panelProps()} />);
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "模型排序" }), "size_desc");
    expect(JSON.parse(window.sessionStorage.getItem("novasight.model-filters.v1") ?? "null").sort).toBe("size_desc");
    first.unmount();
    render(<ModelSelectionPanel {...panelProps()} />);
    expect(screen.getByRole("combobox", { name: "模型排序" })).toHaveValue("size_desc");
  });
});
