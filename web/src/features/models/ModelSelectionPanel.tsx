import { useEffect, useMemo, useState } from "react";

import type {
  ModelArtifact,
  ModelCatalogDirectory,
  ModelCatalogModel,
  ModelRecommendation,
  ParserPresetId,
  ModelVersion
} from "../../api";
import { StatusIndicator } from "../../components/ui";
import { NovaIcon } from "../../components/visual";
import { flattenCatalogModels, ModelCatalogTree } from "./ModelCatalogTree";
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
  onSelectModel: (model: ModelCatalogModel) => void;
  onSaveMetadata: (recommendation: ModelRecommendation, tags: string[]) => void;
  onSwitch: () => void;
}

export function ModelSelectionPanel({
  root,
  loading,
  directoryCount,
  modelCount,
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
  onSelectModel,
  onSaveMetadata,
  onSwitch
}: ModelSelectionPanelProps) {
  const [recommendationFilter, setRecommendationFilter] = useState<ModelRecommendation | "all">("all");
  const [tagFilters, setTagFilters] = useState<Set<string>>(() => new Set());
  const [draftRecommendation, setDraftRecommendation] = useState<ModelRecommendation>("unrated");
  const [draftTags, setDraftTags] = useState<string[]>([]);
  const [newTag, setNewTag] = useState("");
  const models = useMemo(() => flattenCatalogModels(root), [root]);
  const availableTags = useMemo(
    () => Array.from(new Set(models.flatMap((model) => model.tags))).sort((left, right) => left.localeCompare(right)),
    [models]
  );
  const filteredModels = useMemo(
    () => models.filter((model) =>
      (recommendationFilter === "all" || model.recommendation === recommendationFilter) &&
      Array.from(tagFilters).every((tag) => model.tags.includes(tag))
    ),
    [models, recommendationFilter, tagFilters]
  );

  useEffect(() => {
    setDraftRecommendation(selectedModel?.recommendation ?? "unrated");
    setDraftTags(selectedModel?.tags ?? []);
    setNewTag("");
  }, [selectedModel?.recommendation, selectedModel?.relative_path, selectedModel?.tags]);

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
  const metadataDirty = selectedModel !== null && (
    draftRecommendation !== selectedModel.recommendation ||
    draftTags.length !== selectedModel.tags.length ||
    draftTags.some((tag, index) => tag !== selectedModel.tags[index])
  );
  const addTag = (value: string) => {
    const tag = value.trim();
    if (!tag || draftTags.some((item) => item.toLocaleLowerCase() === tag.toLocaleLowerCase())) return;
    setDraftTags((current) => [...current, tag]);
    setNewTag("");
  };
  const recommendationOptions: Array<{ value: ModelRecommendation; label: string }> = [
    { value: "recommended", label: "推荐" },
    { value: "unrated", label: "待整理" },
    { value: "not_recommended", label: "不推荐" }
  ];
  const suggestedTags = ["高精度模型", "低精度模型", "低延迟", "延迟大", "稳定", "实验模型"];

  return (
    <div className="model-selection-panel">
      <header className="model-selection-toolbar">
        <div>
          <strong>模型目录</strong>
          <span>{directoryCount} 个物理文件夹 · 当前显示 {filteredModels.length}/{modelCount} 个模型</span>
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

      <section className="model-vault-filters" aria-label="模型筛选">
        <div className="model-vault-recommendation-filter" role="group" aria-label="推荐状态筛选">
          {([
            ["all", "全部"],
            ["recommended", "推荐"],
            ["unrated", "待整理"],
            ["not_recommended", "不推荐"]
          ] as const).map(([value, label]) => (
            <button
              aria-pressed={recommendationFilter === value}
              className={recommendationFilter === value ? "active" : ""}
              key={value}
              onClick={() => setRecommendationFilter(value)}
              type="button"
            >
              {label}
            </button>
          ))}
        </div>
        <div className="model-vault-tag-filter" aria-label="标签筛选">
          <span>标签筛选</span>
          {availableTags.length > 0 ? availableTags.map((tag) => (
            <button
              aria-pressed={tagFilters.has(tag)}
              className={tagFilters.has(tag) ? "active" : ""}
              key={tag}
              onClick={() => setTagFilters((current) => {
                const next = new Set(current);
                if (next.has(tag)) next.delete(tag); else next.add(tag);
                return next;
              })}
              type="button"
            >
              {tag}
            </button>
          )) : <small>保存标签后可在这里筛选</small>}
          {tagFilters.size > 0 ? (
            <button className="clear" onClick={() => setTagFilters(new Set())} type="button">清除</button>
          ) : null}
        </div>
      </section>

      <div className="model-selection-workspace">
        <div className="model-selection-browser">
          {loading && root === null ? (
            <div className="model-catalog-placeholder">正在读取 models 目录...</div>
          ) : filteredModels.length > 0 ? (
            <ModelCatalogTree
              models={filteredModels}
              selectedPath={selectedPath}
              activeArtifactId={activeArtifactId}
              onSelectModel={onSelectModel}
            />
          ) : (
            <div className="model-catalog-placeholder">
              {models.length > 0 ? "没有符合当前推荐状态与标签的模型。" : "models 目录中没有 .onnx 或 .engine 模型。"}
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

          <section className="model-metadata-editor" aria-labelledby="model-metadata-title">
            <div className="model-metadata-heading">
              <div>
                <strong id="model-metadata-title">整理与标签</strong>
                <small>仅更新模型目录元数据，不读取 Engine，也不会触碰运行中的推理主链。</small>
              </div>
              {metadataDirty ? <span>待保存</span> : null}
            </div>
            <div className="model-recommendation-control" role="group" aria-label="模型推荐状态">
              {recommendationOptions.map((option) => (
                <button
                  aria-pressed={draftRecommendation === option.value}
                  className={draftRecommendation === option.value ? "active" : ""}
                  disabled={busy !== null || selectedModel === null}
                  key={option.value}
                  onClick={() => setDraftRecommendation(option.value)}
                  type="button"
                >
                  {option.label}
                </button>
              ))}
            </div>
            <div className="model-tag-editor">
              <div className="model-tag-list">
                {draftTags.map((tag) => (
                  <button
                    aria-label={`移除标签 ${tag}`}
                    disabled={busy !== null}
                    key={tag}
                    onClick={() => setDraftTags((current) => current.filter((item) => item !== tag))}
                    type="button"
                  >
                    {tag}<span aria-hidden="true">×</span>
                  </button>
                ))}
                {draftTags.length === 0 ? <small>尚未添加标签</small> : null}
              </div>
              <div className="model-tag-input-row">
                <input
                  aria-label="新增模型标签"
                  disabled={busy !== null || selectedModel === null}
                  maxLength={32}
                  onChange={(event) => setNewTag(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      addTag(newTag);
                    }
                  }}
                  placeholder="输入自定义标签"
                  value={newTag}
                />
                <button
                  className="console-button secondary"
                  disabled={busy !== null || selectedModel === null || newTag.trim() === ""}
                  onClick={() => addTag(newTag)}
                  type="button"
                >
                  添加
                </button>
              </div>
              <div className="model-tag-suggestions" aria-label="常用标签">
                {suggestedTags.filter((tag) => !draftTags.includes(tag)).map((tag) => (
                  <button disabled={busy !== null || selectedModel === null} key={tag} onClick={() => addTag(tag)} type="button">
                    + {tag}
                  </button>
                ))}
              </div>
            </div>
            <button
              className="console-button secondary model-metadata-save"
              disabled={busy !== null || !metadataDirty || selectedModel?.kind !== "engine"}
              onClick={() => onSaveMetadata(draftRecommendation, draftTags)}
              type="button"
            >
              {busy === "model.metadata" ? "保存中..." : "保存整理结果"}
            </button>
          </section>
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
