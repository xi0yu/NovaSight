import { useState } from "react";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import type { ModelCatalogModel, ModelCatalogResponse } from "../../api";
import { ModelSwitchDialog } from "../models/ModelSwitchDialog";
import { useModelSwitchWorkflow } from "./useModelSwitchWorkflow";

const api = vi.hoisted(() => ({ registerCatalogModel: vi.fn(), updateModelArtifactMetadata: vi.fn(), getModelCatalog: vi.fn(), publishModel: vi.fn() }));
vi.mock("../../api", async (original) => ({ ...await original<typeof import("../../api")>(), ...api }));
vi.mock("../../lib/toast", () => ({ reportError: vi.fn() }));
const model: ModelCatalogModel = {
  type: "model", kind: "engine", name: "trial.engine", relative_path: "trial.engine",
  size_bytes: 10, scan_status: "ready", scan_reason: "", recommendation: "unrated", tags: [],
};

function Workspace({ registered = false, refreshFails = false }: { registered?: boolean; refreshFails?: boolean }) {
  const [catalog, setCatalog] = useState<ModelCatalogResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const workflow = useModelSwitchWorkflow({
    applyModelCatalogResult: setCatalog,
    onRefresh: async () => { if (refreshFails) throw new Error("刷新失败"); },
    parserPreset: "auto", physicalOutputEnabled: false, runtimeMainlineRunning: false,
    selectedCatalogModel: registered ? { ...model, project_id: 1, artifact_id: 2 } : model,
    setBusy, setConfirmationRequest: () => {}, setLocalError: setError,
    setModelCatalogMessage: () => {}, setModelCatalogRefreshKey: () => {}, setModelDetailsRefreshKey: () => {},
    setModelManagerDialogOpen: () => {}, setSelectedModelArtifactId: () => {}, setSelectedModelProjectId: () => {}, setSelectedModelVersionId: () => {},
  });
  const file = catalog?.root.children.find((node) => node.type === "model");
  return <>
    <button disabled={!!busy} onClick={() => void workflow.saveMetadata("recommended", ["稳定"])}>保存标签</button>
    <button disabled={!!busy} onClick={() => workflow.switchModel()}>部署模型</button>
    {error ? <p role="alert">{error}</p> : null}
    {file?.type === "model" && file.artifact_id ? <p>{file.name} · 已登记</p> : null}
    <ModelSwitchDialog completedStages={workflow.completedStages} currentStage={workflow.stageIndex}
      detail={workflow.progressDetail} error={workflow.dialogError} modelName={model.name}
      onClose={workflow.closeDialog} open={workflow.dialogOpen} status={workflow.dialogStatus} dialogRef={workflow.dialogRef} />
  </>;
}

it("keeps the registered file visible and explains a subsequent tag-write failure", async () => {
  api.registerCatalogModel.mockResolvedValue({ project: { id: 1 }, version: { id: 3 }, artifact: { id: 2 }, created: true });
  api.updateModelArtifactMetadata.mockRejectedValue(new Error("标签写入失败"));
  api.getModelCatalog.mockResolvedValue({ root: { type: "directory", name: "models", relative_path: "", children: [{ ...model, project_id: 1, artifact_id: 2 }] } });
  render(<Workspace />);
  await userEvent.click(screen.getByRole("button", { name: "保存标签" }));
  expect(await screen.findByText("trial.engine · 已登记")).toBeVisible();
  expect(screen.getByRole("alert")).toHaveTextContent("标签写入失败");
  expect(screen.getByRole("button", { name: "保存标签" })).toBeEnabled();
});

it("retains the confirmed deployment when reading the next snapshot fails", async () => {
  let complete: (value: unknown) => void = () => {};
  api.publishModel.mockReturnValue(new Promise((resolve) => { complete = resolve; }));
  render(<Workspace registered refreshFails />);
  await userEvent.dblClick(screen.getByRole("button", { name: "部署模型" }));
  await act(async () => complete({ report: { applied: true, message: "已部署" }, preparation: { manifest_action: "reused" } }));
  const dialog = await screen.findByRole("dialog", { name: "模型切换已确认" });
  expect(dialog).toHaveTextContent("最新状态读取失败");
  expect(screen.getByRole("button", { name: "完成" })).toBeEnabled();
  expect(api.publishModel).toHaveBeenCalledTimes(1);
});
