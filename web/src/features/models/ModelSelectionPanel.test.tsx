import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ModelSelectionPanel } from "./ModelSelectionPanel";

describe("ModelSelectionPanel", () => {
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
});
