import { render, screen, within } from "@testing-library/react";
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
      onParserPresetChange: vi.fn(), onRefresh: vi.fn(), onSelectModel: vi.fn(),
      onSaveMetadata: vi.fn(), onSwitch: vi.fn(), ...overrides,
    };
  }

  it("keeps saved tag filters while the catalog is still loading", () => {
    window.sessionStorage.setItem("novasight.model-filters.v1", JSON.stringify({ recommendation: "all", tags: ["稳定"] }));
    const view = render(<ModelSelectionPanel {...panelProps({ root: null, loading: true })} />);
    view.rerender(<ModelSelectionPanel {...panelProps()} />);
    expect(screen.getByRole("button", { name: "稳定" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText(/当前显示 1\/2 个模型/)).toBeInTheDocument();
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

  it("includes a typed tag when saving and protects an unsaved metadata draft", async () => {
    const onSaveMetadata = vi.fn();
    render(<ModelSelectionPanel {...panelProps({ onSaveMetadata })} />);
    await userEvent.click(screen.getByText("整理模型标签与推荐状态"));
    await userEvent.type(screen.getByRole("textbox", { name: "新增模型标签" }), "低延迟");
    expect(screen.getByRole("button", { name: /other.engine/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: "刷新模型" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "保存整理结果" }));
    expect(onSaveMetadata).toHaveBeenCalledWith("recommended", ["稳定", "低延迟"]);
  });

  it("lets the operator explicitly discard metadata edits before choosing another model", async () => {
    render(<ModelSelectionPanel {...panelProps()} />);
    await userEvent.click(screen.getByText("整理模型标签与推荐状态"));
    await userEvent.click(within(screen.getByRole("group", { name: "模型推荐状态" })).getByRole("button", { name: "不推荐" }));
    expect(screen.getByRole("button", { name: /other.engine/ })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "放弃整理修改" }));
    expect(screen.getByRole("button", { name: /other.engine/ })).toBeEnabled();
  });

  it("does not offer editable metadata for ONNX files that cannot be saved", async () => {
    const onnx = { ...otherModel, kind: "onnx" as const, name: "other.onnx", relative_path: "other.onnx" };
    render(<ModelSelectionPanel {...panelProps({ root: { ...catalog, children: [onnx] }, selectedModel: onnx, selectedPath: onnx.relative_path })} />);
    await userEvent.click(screen.getByText("整理模型标签与推荐状态"));
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
        onSelectModel={vi.fn()}
        onSaveMetadata={vi.fn()}
        onSwitch={vi.fn()}
      />
    );

    expect(screen.queryByRole("region", { name: "模型筛选" })).not.toBeInTheDocument();
    expect(screen.getByText(/将 \.engine 文件放入设备的 models 目录/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "刷新模型" })).toBeEnabled();
    expect(screen.getByRole("list", { name: "切换模型的三个步骤" })).toHaveTextContent("确认结果");
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
      parserPreset="auto" onParserPresetChange={vi.fn()} onRefresh={vi.fn()}
      onSelectModel={vi.fn()} onSaveMetadata={vi.fn()} onSwitch={onSwitch}
    />);

    expect(screen.getByText(/所选文件尚未生效/)).toBeVisible();
    expect(screen.getByRole("combobox", { name: "模型解析格式" })).not.toBeVisible();
    await userEvent.click(screen.getByText("查看文件详情与解析设置"));
    expect(screen.getByRole("combobox", { name: "模型解析格式" })).toBeVisible();
    expect(onSwitch).not.toHaveBeenCalled();
  });
});
