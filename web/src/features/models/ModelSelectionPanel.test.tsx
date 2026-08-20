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
    expect(screen.getByText(/将 \.onnx 或 \.engine 文件放入设备的 models 目录/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "刷新模型" })).toBeEnabled();
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
});
