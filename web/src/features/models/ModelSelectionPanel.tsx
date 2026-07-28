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

export interface ModelSelectionPanelProps {
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
  activeArtifactPath: string;
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
  activeArtifactPath,
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
    "切换时读取真实契约";
  const switchLabel = busy === "model.switch"
    ? "正在验证并切换..."
    : "验证并切换到所选模型";

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
              <span>所选 Engine 文件</span>
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
              <dt>当前使用路径</dt>
              <dd title={activeArtifactPath}>{activeArtifactPath || "未加载产物"}</dd>
            </div>
            <div className="wide">
              <dt>所选文件路径</dt>
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
              <dd>{selectedArtifact ? modelStatusLabel(selectedArtifact.status) : selectedModel?.kind === "engine" ? "后缀已接受，切换时验证" : "不可加载"}</dd>
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
                <small>目录浏览不读取 TensorRT；确认切换后才验证真实 tensor 契约。</small>
              </dd>
            </div>
          </dl>
        </aside>
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

      {selectedModel?.kind === "engine" ? (
        <p className="console-field-hint">
          点击后会打开切换进度：验证 Engine I/O、复用或生成 manifest，再应用运行态。
        </p>
      ) : null}

    </div>
  );
}
