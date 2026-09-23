import {
  useCallback,
  useEffect,
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
import { acquireBodyScrollLock, releaseBodyScrollLock, trapDialogTabKey } from "./dialogFocus";

const MODEL_SWITCH_STAGE_COUNT = 5;

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

function parserOutputFamilyLabel(outputFamily: string | undefined): string {
  if (outputFamily === "yolov5") {
    return "YOLO v5 解析格式";
  }
  if (outputFamily === "yolov8_yolo11") {
    return "YOLO v8 / v11 解析格式";
  }
  return "";
}

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

  useEffect(() => {
    if (!dialogOpen) {
      return undefined;
    }
    acquireBodyScrollLock();
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    window.requestAnimationFrame(() => dialogRef.current?.focus());
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && dialogStatus !== "running") {
        setDialogOpen(false);
      } else {
        trapDialogTabKey(event, dialogRef.current);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      releaseBodyScrollLock();
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [dialogOpen, dialogStatus]);

  const ensureCatalogModelRegistration = useCallback(async (model: ModelCatalogModel) => {
    if (typeof model.project_id === "number" && typeof model.artifact_id === "number") {
      return { projectId: model.project_id, artifactId: model.artifact_id, created: false };
    }
    const registered = await registerCatalogModel(model.relative_path);
    setSelectedModelProjectId(registered.project.id);
    setSelectedModelVersionId(registered.version.id);
    setSelectedModelArtifactId(registered.artifact.id);
    setModelCatalogMessage(`已登记模型引用：${model.relative_path}；Engine 文件保持原位。`);
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
          detail = "；模型列表也未能重新读取，请点击“刷新模型”核对登记状态";
        }
      }
      setLocalError(metadataSaved
        ? `模型标签已保存，但页面更新失败：${getErrorMessage(err)}。请点击“刷新模型”核对。`
        : registered
          ? `模型引用已登记，标签保存结果未确认：${getErrorMessage(err)}${detail}。当前整理输入已保留。`
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

  const performSwitch = useCallback(async (physicalOutputAcknowledged = false) => {
    const model = selectedCatalogModel;
    if (!model || model.kind !== "engine") {
      setLocalError("请选择 TensorRT engine 产物。");
      return;
    }
    setBusy("model.switch");
    setLocalError(null);
    setMessage("");
    setDialogOpen(true);
    setDialogStatus("running");
    setStageIndex(0);
    setCompletedStages(0);
    setDialogError("");
    setProgressDetail("已按 .engine 后缀接受候选，准备登记模型引用。");
    try {
      setCompletedStages(1);
      setStageIndex(1);
      if (typeof model.project_id !== "number" || typeof model.artifact_id !== "number") {
        setProgressDetail("模型尚未登记，正在建立轻量文件引用；此步骤不会读取 Engine 内容。");
      } else {
        setProgressDetail("已找到现有模型登记，跳过重复登记。");
      }
      const { projectId, artifactId } = await ensureCatalogModelRegistration(model);
      setCompletedStages(2);
      setStageIndex(2);
      setProgressDetail("正在验证模型输入输出，并准备运行配置。");
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
      const parserLabel = parserOutputFamilyLabel(response.parser_contract?.output_family);
      const switchSummary = activeNoOp
        ? "所选模型已是当前模型，运行态保持不变"
        : response.report.message;
      const manifestSummary = response.preparation.manifest_action === "generated"
        ? "已自动生成运行配置"
        : response.preparation.manifest_action === "reused"
          ? "已复用匹配的运行配置"
          : "运行配置保持不变";
      setMessage(
        parserLabel
          ? `${switchSummary} · 已验证 ${parserLabel} · NovaSight 内置解析器`
          : switchSummary
      );
      setCompletedStages(MODEL_SWITCH_STAGE_COUNT);
      setStageIndex(MODEL_SWITCH_STAGE_COUNT - 1);
      setDialogStatus("success");
      setProgressDetail(`${manifestSummary}；${switchSummary}`);
      setModelManagerDialogOpen(false);
      setModelCatalogRefreshKey((current) => current + 1);
      setModelDetailsRefreshKey((current) => current + 1);
      await onRefresh();
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      setDialogStatus("failed");
      setDialogError(`${errorMessage}。请核对页面上的当前模型和运行状态；请求失败或回执解析失败时，不能假定切换未生效。`);
      setLocalError(`模型切换结果未确认：${errorMessage}`);
      reportError(err, { source: "studio", title: "操作失败" });
      await onRefresh();
    } finally {
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
    const candidatePath = model.relative_path;
    const needsRegistration = typeof model.project_id !== "number" || typeof model.artifact_id !== "number";
    setConfirmationRequest({
      eyebrow: runtimeMainlineRunning ? "运行中切换模型" : "部署模型",
      title: runtimeMainlineRunning ? "停止并重启推理主链？" : "验证并部署所选模型？",
      description: runtimeMainlineRunning
        ? "当前主链正在运行。模型发布会停止现有管线、验证所选 Engine，并在成功后使用新模型重新启动。"
        : "将验证所选 Engine 并保存为当前模型；主链保持停止，下一次启动时使用它。",
      details: [
        `所选模型：${candidatePath}`,
        ...(needsRegistration ? ["首次使用会登记此 Engine 的路径引用；登记后不能直接移动或改名文件。"] : []),
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
