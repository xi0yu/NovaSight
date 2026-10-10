import {
  useCallback,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction
} from "react";

import {
  getModelCatalog,
  publishModel,
  registerCatalogModel,
  updateModelArtifactMetadata,
  type ModelCatalogModel,
  type ModelCatalogResponse,
  type ModelRecommendation,
  type ModelPublishResponse,
  type ParserPresetId
} from "../../api";
import type { ModelSwitchDialogStatus } from "../models/ModelSwitchDialog";
import { reportError } from "../../lib/toast";
import { getErrorMessage } from "../shared/format";
import type { ActionConfirmationRequest } from "./ActionConfirmationDialog";

const MODEL_SWITCH_STAGE_COUNT = 3;

export function isActiveModelNoOp(
  response: ModelPublishResponse,
  requestedArtifactId: number,
): boolean {
  return response.report.applied === false
    && response.report.rolled_back === false
    && response.report.artifact_id === requestedArtifactId
    && response.deployment.artifact_id === requestedArtifactId;
}

type UseModelSwitchWorkflowInput = {
  applyModelCatalogResult: (result: ModelCatalogResponse) => void;
  onRefresh: () => Promise<void>;
  parserPreset: ParserPresetId;
  physicalOutputEnabled: boolean;
  runtimeMainlineRunning: boolean;
  selectedCatalogModel: ModelCatalogModel | null;
  setBusy: (busy: string | null) => void;
  setConfirmationRequest: Dispatch<SetStateAction<ActionConfirmationRequest | null>>;
  setLocalError: (message: string | null) => void;
  setModelCatalogMessage: (message: string) => void;
  setModelCatalogRefreshKey: Dispatch<SetStateAction<number>>;
  setModelDetailsRefreshKey: Dispatch<SetStateAction<number>>;
  setModelManagerDialogOpen: (open: boolean) => void;
  setSelectedModelArtifactId: (id: number | "") => void;
  setSelectedModelProjectId: (id: number | "") => void;
  setSelectedModelVersionId: (id: number | "") => void;
};

export function useModelSwitchWorkflow({
  applyModelCatalogResult,
  onRefresh,
  parserPreset,
  physicalOutputEnabled,
  runtimeMainlineRunning,
  selectedCatalogModel,
  setBusy,
  setConfirmationRequest,
  setLocalError,
  setModelCatalogMessage,
  setModelCatalogRefreshKey,
  setModelDetailsRefreshKey,
  setModelManagerDialogOpen,
  setSelectedModelArtifactId,
  setSelectedModelProjectId,
  setSelectedModelVersionId
}: UseModelSwitchWorkflowInput) {
  const [message, setMessage] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogStatus, setDialogStatus] = useState<ModelSwitchDialogStatus>("running");
  const [stageIndex, setStageIndex] = useState(0);
  const [completedStages, setCompletedStages] = useState(0);
  const [progressDetail, setProgressDetail] = useState("");
  const [dialogError, setDialogError] = useState("");
  const dialogRef = useRef<HTMLElement | null>(null);

  const closeDialog = useCallback(() => {
    setDialogOpen(false);
  }, []);

  const ensureCatalogModelRegistration = useCallback(async (model: ModelCatalogModel) => {
    if (typeof model.project_id === "number" && typeof model.artifact_id === "number") {
      return { projectId: model.project_id, artifactId: model.artifact_id, created: false };
    }
    const registered = await registerCatalogModel(model.relative_path);
    setSelectedModelProjectId(registered.project.id);
    setSelectedModelVersionId(registered.version.id);
    setSelectedModelArtifactId(registered.artifact.id);
    setModelCatalogMessage(`模型已加入列表：${model.relative_path}；文件保留在原位置。`);
    return {
      artifactId: registered.artifact.id,
      created: registered.created,
      projectId: registered.project.id
    };
  }, [
    setModelCatalogMessage,
    setSelectedModelArtifactId,
    setSelectedModelProjectId,
    setSelectedModelVersionId
  ]);

  const saveMetadata = useCallback(async (
    recommendation: ModelRecommendation,
    tags: string[]
  ) => {
    const model = selectedCatalogModel;
    if (!model || model.kind !== "engine") {
      setLocalError("请选择 TensorRT engine 模型后再整理标签。");
      return;
    }
    setBusy("model.metadata");
    setLocalError(null);
    let registered = false;
    let metadataSaved = false;
    try {
      const registration = await ensureCatalogModelRegistration(model);
      registered = true;
      await updateModelArtifactMetadata(registration.artifactId, recommendation, tags);
      metadataSaved = true;
      const updatedCatalog = await getModelCatalog(false);
      applyModelCatalogResult(updatedCatalog);
      setModelCatalogMessage(`已保存 ${model.name} 的推荐状态与 ${tags.length} 个标签。`);
      if (registration.created) {
        setModelDetailsRefreshKey((current) => current + 1);
        await onRefresh();
      }
    } catch (err) {
      let detail = "；目录已重新读取，请核对标签后重试";
      if (registered && !metadataSaved) {
        try {
          applyModelCatalogResult(await getModelCatalog(false));
        } catch {
          detail = "；模型列表也未能重新读取，请点击“刷新模型”核对文件状态";
        }
      }
      setLocalError(metadataSaved
        ? `模型标签已保存，但页面更新失败：${getErrorMessage(err)}。请点击“刷新模型”核对。`
        : registered
          ? `模型已加入列表，标签保存结果未确认：${getErrorMessage(err)}${detail}。当前整理输入已保留。`
          : `模型整理结果保存失败：${getErrorMessage(err)}`);
      reportError(err, { source: "model-metadata", title: "模型整理失败" });
    } finally {
      setBusy(null);
    }
  }, [
    applyModelCatalogResult,
    ensureCatalogModelRegistration,
    onRefresh,
    selectedCatalogModel,
    setBusy,
    setLocalError,
    setModelCatalogMessage,
    setModelDetailsRefreshKey
  ]);

  const switchingRef = useRef(false);
  const performSwitch = useCallback(async (physicalOutputAcknowledged = false) => {
    if (switchingRef.current) return;
    const model = selectedCatalogModel;
    if (!model || model.kind !== "engine") {
      setLocalError("请选择 TensorRT engine 产物。");
      return;
    }
    switchingRef.current = true;
    setBusy("model.switch");
    setLocalError(null);
    setMessage("");
    setDialogOpen(true);
    setDialogStatus("running");
    setStageIndex(0);
    setCompletedStages(0);
    setDialogError("");
    setProgressDetail("已选择模型文件，正在准备使用。");
    try {
      setCompletedStages(1);
      setStageIndex(1);
      if (typeof model.project_id !== "number" || typeof model.artifact_id !== "number") {
        setProgressDetail("正在整理所选模型，随后由设备检查兼容性。");
      } else {
        setProgressDetail("所选模型已准备，正在交给设备检查。");
      }
      const { projectId, artifactId } = await ensureCatalogModelRegistration(model);
      setCompletedStages(2);
      setStageIndex(2);
      setProgressDetail("等待设备完成验证与切换。设备未提供分阶段进度，请勿重复操作。");
      const response = await publishModel(
        projectId,
        artifactId,
        parserPreset,
        physicalOutputAcknowledged
      );
      const activeNoOp = isActiveModelNoOp(response, artifactId);
      if (!response.report.applied && !activeNoOp) {
        throw new Error(response.report.message);
      }
      const switchSummary = activeNoOp
        ? "所选模型已在使用，保持当前运行状态"
        : "模型已切换，可在设备页查看识别画面";
      const manifestSummary = "模型设置已准备";
      setMessage(switchSummary);
      setCompletedStages(MODEL_SWITCH_STAGE_COUNT);
      setStageIndex(MODEL_SWITCH_STAGE_COUNT - 1);
      setDialogStatus("success");
      setProgressDetail(`${manifestSummary}；${switchSummary}`);
      setModelManagerDialogOpen(false);
      setModelCatalogRefreshKey((current) => current + 1);
      setModelDetailsRefreshKey((current) => current + 1);
      try {
        await onRefresh();
      } catch (refreshError) {
        setProgressDetail(`${manifestSummary}；${switchSummary}。切换已确认，但最新状态读取失败，请刷新页面核对。`);
        reportError(refreshError, { source: "model-refresh", title: "模型已切换，状态读取失败", popup: false });
      }
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      setDialogStatus("failed");
      setDialogError(`${errorMessage}。请核对页面上的当前模型和运行状态；请求失败或回执解析失败时，不能假定切换未生效。`);
      reportError(err, { source: "model-switch", title: "模型切换结果未确认", popup: false });
      try { await onRefresh(); } catch { /* Keep the original operation error visible in the dialog. */ }
    } finally {
      switchingRef.current = false;
      setBusy(null);
    }
  }, [
    ensureCatalogModelRegistration,
    onRefresh,
    parserPreset,
    selectedCatalogModel,
    setBusy,
    setLocalError,
    setModelCatalogRefreshKey,
    setModelDetailsRefreshKey,
    setModelManagerDialogOpen
  ]);

  const switchModel = useCallback(() => {
    const model = selectedCatalogModel;
    if (!model || model.kind !== "engine") {
      setLocalError("请选择 TensorRT engine 产物。");
      return;
    }
    if (!runtimeMainlineRunning && !physicalOutputEnabled && typeof model.project_id === "number" && typeof model.artifact_id === "number") {
      void performSwitch(false);
      return;
    }
    const candidatePath = model.relative_path;
    const needsRegistration = typeof model.project_id !== "number" || typeof model.artifact_id !== "number";
    setConfirmationRequest({
      eyebrow: runtimeMainlineRunning ? "运行中切换模型" : "部署模型",
      title: runtimeMainlineRunning ? "停止并重启推理主链？" : "验证并部署所选模型？",
      description: runtimeMainlineRunning
        ? "当前主链正在运行。切换会停止当前识别，检查模型兼容性，成功后用新模型恢复识别。"
        : "将检查并保存所选模型；设备保持停止，下次启动时使用它。",
      details: [
        `所选模型：${candidatePath}`,
        ...(needsRegistration ? ["使用后请保留模型文件的位置与名称，否则下次无法找到它。"] : []),
        physicalOutputEnabled
          ? "当前物理输出已开启。确认后，切换或失败回滚可能重新开启设备输出；请确保设备有人看管。"
          : runtimeMainlineRunning
            ? "切换期间识别结果会短暂停止；失败时会回滚原部署。"
            : "此操作不会启动主链或打开物理输出；失败时保留原部署。"
      ],
      confirmLabel: runtimeMainlineRunning ? "确认切换模型" : "确认验证并部署",
      danger: runtimeMainlineRunning || physicalOutputEnabled,
      handoffOnConfirm: true,
      onConfirm: () => performSwitch(physicalOutputEnabled)
    });
  }, [
    performSwitch,
    physicalOutputEnabled,
    runtimeMainlineRunning,
    selectedCatalogModel,
    setConfirmationRequest,
    setLocalError
  ]);

  return {
    closeDialog,
    completedStages,
    dialogError,
    dialogOpen,
    dialogRef,
    dialogStatus,
    message,
    progressDetail,
    saveMetadata,
    stageIndex,
    switchModel
  };
}
