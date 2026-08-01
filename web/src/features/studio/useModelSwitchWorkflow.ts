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
  type ParserPresetId
} from "../../api";
import type { ModelSwitchDialogStatus } from "../models/ModelSwitchDialog";
import { reportError } from "../../lib/toast";
import { getErrorMessage } from "../shared/format";
import type { ActionConfirmationRequest } from "./ActionConfirmationDialog";
import { trapDialogTabKey } from "./dialogFocus";

const MODEL_SWITCH_STAGE_COUNT = 5;

type UseModelSwitchWorkflowInput = {
  applyModelCatalogResult: (result: ModelCatalogResponse) => void;
  onRefresh: () => Promise<void>;
  parserPreset: ParserPresetId;
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

function parserCompatibilityLabel(compatibility: string | undefined): string {
  if (compatibility === "yolov5") {
    return "YOLO v5 兼容";
  }
  if (compatibility === "yolov8_yolo11") {
    return "YOLO v8 / v11 兼容";
  }
  return "";
}

export function useModelSwitchWorkflow({
  applyModelCatalogResult,
  onRefresh,
  parserPreset,
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
    const previousOverflow = document.body.style.overflow;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    document.body.style.overflow = "hidden";
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
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
      previousFocus?.focus();
    };
  }, [dialogOpen, dialogStatus]);

  const ensureCatalogModelRegistration = useCallback(async (model: ModelCatalogModel) => {
    if (typeof model.project_id === "number" && typeof model.artifact_id === "number") {
      return { projectId: model.project_id, artifactId: model.artifact_id };
    }
    const registered = await registerCatalogModel(model.relative_path);
    setSelectedModelProjectId(registered.project.id);
    setSelectedModelVersionId(registered.version.id);
    setSelectedModelArtifactId(registered.artifact.id);
    setModelCatalogMessage(`已登记模型引用：${model.relative_path}；Engine 文件保持原位。`);
    return { projectId: registered.project.id, artifactId: registered.artifact.id };
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
    try {
      const { artifactId } = await ensureCatalogModelRegistration(model);
      await updateModelArtifactMetadata(artifactId, recommendation, tags);
      const updatedCatalog = await getModelCatalog(false);
      applyModelCatalogResult(updatedCatalog);
      setModelCatalogMessage(`已保存 ${model.name} 的推荐状态与 ${tags.length} 个标签。`);
      setModelDetailsRefreshKey((current) => current + 1);
      await onRefresh();
    } catch (err) {
      setLocalError(`模型整理结果保存失败：${getErrorMessage(err)}`);
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

  const performSwitch = useCallback(async () => {
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
      setProgressDetail("正在后端事务中验证 TensorRT 契约，并复用或生成运行 manifest。");
      const response = await publishModel(
        projectId,
        artifactId,
        parserPreset
      );
      if (response.report && !response.report.applied) {
        throw new Error(response.report.message);
      }
      const parserLabel = parserCompatibilityLabel(response.parser_contract?.compatibility);
      const switchSummary = response.report?.message ??
        "Engine 契约读取完成，DeepStream 配置已自动生成并切换。";
      const manifestSummary = response.preparation?.manifest_action === "generated"
        ? "已自动生成运行 manifest"
        : response.preparation?.manifest_action === "reused"
          ? "已复用匹配的运行 manifest"
          : "运行 manifest 已准备";
      setMessage(
        parserLabel
          ? `${switchSummary} · 已验证 ${parserLabel} · NovaSight 内置 parser`
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
      setDialogError(errorMessage);
      setLocalError(`模型切换未生效：${errorMessage}`);
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
    if (!runtimeMainlineRunning) {
      void performSwitch();
      return;
    }
    const candidatePath = model.relative_path;
    setConfirmationRequest({
      eyebrow: "运行中切换模型",
      title: "停止并重启推理主链？",
      description: "当前主链正在运行。模型发布会停止现有管线、验证所选 Engine，并在成功后使用新模型重新启动。",
      details: [
        `所选模型：${candidatePath}`,
        "切换期间 DetectionBatch 与物理输出会短暂停止；失败时后端会回滚原部署。"
      ],
      confirmLabel: "确认切换模型",
      danger: true,
      handoffOnConfirm: true,
      onConfirm: performSwitch
    });
  }, [
    performSwitch,
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
