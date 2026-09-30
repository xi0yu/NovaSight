import { act, renderHook, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";

import type { ModelCatalogModel, ModelCatalogResponse } from "../../api";
import type { ActionConfirmationRequest } from "./ActionConfirmationDialog";
import { useModelSwitchWorkflow } from "./useModelSwitchWorkflow";

const api = vi.hoisted(() => ({
  registerCatalogModel: vi.fn(),
  updateModelArtifactMetadata: vi.fn(),
  getModelCatalog: vi.fn(),
  publishModel: vi.fn(),
}));
vi.mock("../../api", async (original) => ({
  ...await original<typeof import("../../api")>(),
  ...api,
}));
vi.mock("../../lib/toast", () => ({ reportError: vi.fn() }));

it("reads back a newly registered model when saving its tags fails", async () => {
  const model = {
    type: "model", kind: "engine", name: "trial.engine", relative_path: "trial.engine",
    size_bytes: 10, scan_status: "ready", scan_reason: "", recommendation: "unrated", tags: [],
  } as ModelCatalogModel;
  const catalog = { root: { type: "directory", name: "models", relative_path: "", children: [{ ...model, project_id: 1, artifact_id: 2 }] } } as ModelCatalogResponse;
  api.registerCatalogModel.mockResolvedValue({ project: { id: 1 }, version: { id: 3 }, artifact: { id: 2 }, created: true });
  api.updateModelArtifactMetadata.mockRejectedValue(new Error("标签写入失败"));
  api.getModelCatalog.mockResolvedValue(catalog);
  const applyModelCatalogResult = vi.fn();
  const setLocalError = vi.fn();
  const { result } = renderHook(() => useModelSwitchWorkflow({
    applyModelCatalogResult, onRefresh: vi.fn(async () => {}), parserPreset: "auto",
    physicalOutputEnabled: false, runtimeMainlineRunning: false, selectedCatalogModel: model,
    setBusy: vi.fn(), setConfirmationRequest: vi.fn(), setLocalError,
    setModelCatalogMessage: vi.fn(), setModelCatalogRefreshKey: vi.fn(),
    setModelDetailsRefreshKey: vi.fn(), setModelManagerDialogOpen: vi.fn(),
    setSelectedModelArtifactId: vi.fn(), setSelectedModelProjectId: vi.fn(),
    setSelectedModelVersionId: vi.fn(),
  }));

  await act(async () => { await result.current.saveMetadata("recommended", ["稳定"]); });
  await waitFor(() => expect(applyModelCatalogResult).toHaveBeenCalledWith(catalog));
  expect(setLocalError).toHaveBeenCalledWith(expect.stringContaining("已登记"));
  expect(setLocalError).toHaveBeenCalledWith(expect.stringContaining("标签写入失败"));
});

it("keeps the confirmed publish result if only refreshing the page fails", async () => {
  api.publishModel.mockResolvedValue({ report: { applied: true, message: "已部署" }, preparation: { manifest_action: "reused" } });
  const setConfirmationRequest = vi.fn();
  const { result } = renderHook(() => useModelSwitchWorkflow({
    applyModelCatalogResult: vi.fn(), onRefresh: vi.fn(async () => { throw new Error("刷新失败"); }), parserPreset: "auto",
    physicalOutputEnabled: false, runtimeMainlineRunning: false,
    selectedCatalogModel: { type: "model", name: "demo.engine", relative_path: "demo.engine", kind: "engine", size_bytes: 1, scan_status: "ready", scan_reason: "", recommendation: "unrated", tags: [], project_id: 1, artifact_id: 2 },
    setBusy: vi.fn(), setConfirmationRequest, setLocalError: vi.fn(), setModelCatalogMessage: vi.fn(), setModelCatalogRefreshKey: vi.fn(), setModelDetailsRefreshKey: vi.fn(), setModelManagerDialogOpen: vi.fn(), setSelectedModelArtifactId: vi.fn(), setSelectedModelProjectId: vi.fn(), setSelectedModelVersionId: vi.fn(),
  }));
  act(() => result.current.switchModel());
  const confirmation = setConfirmationRequest.mock.calls[0][0] as ActionConfirmationRequest;
  await act(async () => { await confirmation.onConfirm(); });
  expect(result.current.dialogStatus).toBe("success");
  expect(result.current.progressDetail).toContain("最新状态读取失败");
  expect(result.current.completedStages).toBe(3);
});
