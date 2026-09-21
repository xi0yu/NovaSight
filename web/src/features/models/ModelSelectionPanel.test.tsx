import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelCatalogDirectory } from "../../api";

import { ModelSelectionPanel } from "./ModelSelectionPanel";

describe("ModelSelectionPanel", () => {
  beforeEach(() => window.sessionStorage.clear());

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
