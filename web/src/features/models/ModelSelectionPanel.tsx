import type {
  ModelArtifact,
  ModelCatalogDirectory,
  ModelCatalogModel,
  ParserPresetId,
  ModelVersion
} from "../../api";
import { StatusIndicator } from "../../components/ui";
import { NovaIcon } from "../../components/visual";
import { ModelCatalogTree } from "./ModelCatalogTree";
import {
  artifactStatus,
  formatModelSize,
  modelStatusLabel,
  modelStatusTone
} from "./modelPresentation";

interface ModelSelectionPanelProps {
  root: ModelCatalogDirectory | null;
  loading: boolean;
  directoryCount: number;
  modelCount: number;
  expandedDirectories: Set<string>;
  selectedPath: string | undefined;
  selectedModel: ModelCatalogModel | null;
  selectedArtifact: ModelArtifact | null;
  selectedVersion: ModelVersion | null;
  activeArtifactId: number | null;
  activeModelName: string;
  activeArtifactLabel: string;
  runtimeBackend: string;
  runtimeInputShape: string;
  catalogMessage: string;
  switchMessage: string;
  busy: string | null;
  canSwitch: boolean;
  parserPreset: ParserPresetId;
  onParserPresetChange: (preset: ParserPresetId) => void;
  onRefresh: () => void;
  onToggleDirectory: (relativePath: string) => void;
  onSelectModel: (model: ModelCatalogModel) => void;
  onSwitch: () => void;
}

export function ModelSelectionPanel({
  root,
  loading,
  directoryCount,
  modelCount,
  expandedDirectories,
  selectedPath,
  selectedModel,
  selectedArtifact,
  selectedVersion,
  activeArtifactId,
  activeModelName,
  activeArtifactLabel,
  runtimeBackend,
  runtimeInputShape,
  catalogMessage,
  switchMessage,
  busy,
  canSwitch,
  parserPreset,
  onParserPresetChange,
  onRefresh,
  onToggleDirectory,
  onSelectModel,
  onSwitch
}: ModelSelectionPanelProps) {
  const selectedStatus = selectedModel
    ? selectedModel.artifact_status ?? selectedModel.scan_status
    : artifactStatus(selectedArtifact);
  const switchNeedsPreparation =
    (selectedModel?.kind === "engine" && !selectedArtifact) ||
    selectedArtifact?.status === "pending" ||
    selectedArtifact?.status === "failed";
  const selectedKind = selectedModel?.kind ?? selectedArtifact?.kind;
  const selectedIsActive = selectedArtifact?.id === activeArtifactId;
  const previewBackend = selectedKind === "engine"
    ? "TensorRT / DeepStream"
    : selectedKind === "onnx"
      ? "ONNX（当前主链不可用）"
      : selectedIsActive && runtimeBackend
        ? runtimeBackend
        : "按模型自动选择";
  const previewInputShape = selectedVersion?.input_shape ||
    (selectedIsActive ? runtimeInputShape : "") ||
    "加载后读取真实契约";
  const switchLabel = busy === "model.switch"
    ? "自动配置并切换中..."
    : switchNeedsPreparation
      ? "自动配置并加载模型"
      : "切换到所选模型";

  return (
    <div className="model-selection-panel">
      <header className="model-selection-toolbar">
        <div>
          <strong>模型目录</strong>
          <span>{directoryCount} 个文件夹 · {modelCount} 个模型</span>
        </div>
        <button
          className="console-button secondary"
          disabled={busy !== null}
          onClick={onRefresh}
          type="button"
        >
          <NovaIcon name="refresh" size={15} />
          {busy === "model.refresh" ? "刷新中..." : "刷新模型"}
        </button>
      </header>

      {catalogMessage ? <div className="model-switch-note good">{catalogMessage}</div> : null}

      <div className="model-selection-workspace">
        <div className="model-selection-browser">
          {loading ? (
            <div className="model-catalog-placeholder">正在读取 models 目录...</div>
          ) : root && root.children.length > 0 ? (
            <ModelCatalogTree
              root={root}
              expandedDirectories={expandedDirectories}
              selectedPath={selectedPath}
              activeArtifactId={activeArtifactId}
              onToggleDirectory={onToggleDirectory}
              onSelectModel={onSelectModel}
            />
          ) : (
            <div className="model-catalog-placeholder">
              models 目录中没有 .onnx 或 .engine 模型。
            </div>
          )}
        </div>

        <aside className="model-selection-preview" aria-live="polite">
          <div className="model-selection-preview-heading">
            <span className="model-selection-preview-icon" aria-hidden="true">
              <NovaIcon name="engine" size={20} />
            </span>
            <div>
              <span>所选模型快速预览</span>
              <strong title={selectedModel?.name ?? selectedArtifact?.path ?? ""}>
                {selectedModel?.name ?? selectedArtifact?.path ?? "尚未选择模型"}
              </strong>
            </div>
            <StatusIndicator tone={modelStatusTone(selectedStatus)}>
              {modelStatusLabel(selectedStatus)}
            </StatusIndicator>
          </div>

          <dl className="model-selection-facts">
            <div className="wide">
              <dt>文件路径</dt>
              <dd title={selectedModel?.relative_path ?? selectedArtifact?.path ?? ""}>
                {selectedModel?.relative_path ?? selectedArtifact?.path ?? "-"}
              </dd>
            </div>
            <div>
              <dt>文件类型</dt>
              <dd>{(selectedModel?.kind ?? selectedArtifact?.kind ?? "-").toUpperCase()}</dd>
            </div>
            <div>
              <dt>文件大小</dt>
              <dd>{formatModelSize(selectedModel?.size_bytes ?? selectedArtifact?.size_bytes)}</dd>
            </div>
            <div>
              <dt>模型版本</dt>
              <dd>{selectedVersion?.version === "default" ? "自动发现版本" : selectedVersion?.version ?? "-"}</dd>
            </div>
            <div>
              <dt>输入尺寸</dt>
              <dd>{previewInputShape}</dd>
            </div>
            <div>
              <dt>运行后端</dt>
              <dd>{previewBackend}</dd>
            </div>
            <div>
              <dt>切换准备</dt>
              <dd>{selectedArtifact ? modelStatusLabel(selectedArtifact.status) : selectedModel ? "选择后自动登记" : "-"}</dd>
            </div>
            <div className="wide model-parser-preset">
              <dt>解析兼容模式</dt>
              <dd>
                <select
                  aria-label="模型解析兼容模式"
                  disabled={busy !== null}
                  onChange={(event) => onParserPresetChange(event.target.value as ParserPresetId)}
                  value={parserPreset}
                >
                  <option value="auto">自动识别（推荐）</option>
                  <option value="yolov5">YOLO v5 兼容</option>
                  <option value="yolov8">YOLO v8 兼容</option>
                  <option value="yolo11">YOLO v11 兼容</option>
                  <option value="novasight_generic">NovaSight 通用解析器（内置）</option>
                </select>
                <small>全部使用 NovaSight 内置 parser；选择项会在加载前校验真实 tensor 契约。</small>
              </dd>
            </div>
          </dl>
        </aside>
      </div>

      <div className="model-active-summary">
        <span>当前运行模型</span>
        <b title={activeModelName}>{activeModelName}</b>
        <small title={activeArtifactLabel}>{activeArtifactLabel}</small>
      </div>

      {switchMessage ? <div className="model-switch-note good">{switchMessage}</div> : null}

      <button
        className="console-button primary console-full-button"
        disabled={busy !== null || !canSwitch}
        onClick={onSwitch}
        type="button"
      >
        <NovaIcon name="model-switch" size={16} />
        {switchLabel}
      </button>

      {switchNeedsPreparation ? (
        <p className="console-field-hint">
          后端会读取 Engine I/O、Shape 和数据类型，并自动生成唯一 DeepStream manifest。
        </p>
      ) : null}

    </div>
  );
}
