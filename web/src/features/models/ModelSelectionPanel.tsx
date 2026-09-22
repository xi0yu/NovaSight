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

const RECOMMENDATION_OPTIONS: Array<{ value: ModelRecommendation; label: string }> = [
  { value: "recommended", label: "推荐" },
  { value: "unrated", label: "待整理" },
  { value: "not_recommended", label: "不推荐" }
];
const SUGGESTED_MODEL_TAGS = ["高精度模型", "低精度模型", "低延迟", "延迟大", "稳定", "实验模型"];
const MODEL_FILTER_SESSION_KEY = "novasight.model-filters.v1";

function readModelFilterSession(): {
  recommendation: ModelRecommendation | "all";
  tags: string[];
} {
  try {
    const value: unknown = JSON.parse(window.sessionStorage.getItem(MODEL_FILTER_SESSION_KEY) ?? "null");
    if (!value || typeof value !== "object") return { recommendation: "all", tags: [] };
    const record = value as Record<string, unknown>;
    const recommendation = ["all", "recommended", "unrated", "not_recommended"].includes(String(record.recommendation))
      ? record.recommendation as ModelRecommendation | "all"
      : "all";
    const tags = Array.isArray(record.tags)
      ? record.tags.filter((tag): tag is string => typeof tag === "string")
      : [];
    return { recommendation, tags };
  } catch {
    return { recommendation: "all", tags: [] };
  }
}

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
  activeLoaded?: boolean;
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
  activeLoaded = false,
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
  const [initialFilters] = useState(readModelFilterSession);
  const [recommendationFilter, setRecommendationFilter] = useState<ModelRecommendation | "all">(
    initialFilters.recommendation
  );
  const [tagFilters, setTagFilters] = useState<Set<string>>(() => new Set(initialFilters.tags));
  const [searchTerm, setSearchTerm] = useState("");
  const [draftRecommendation, setDraftRecommendation] = useState<ModelRecommendation>("unrated");
  const [draftTags, setDraftTags] = useState<string[]>([]);
  const [newTag, setNewTag] = useState("");
  const models = useMemo(() => flattenCatalogModels(root), [root]);
  const availableTags = useMemo(
    () => Array.from(new Set(models.flatMap((model) => model.tags))).sort((left, right) => left.localeCompare(right)),
    [models]
  );
  const availableTagSet = useMemo(() => new Set(availableTags), [availableTags]);
  const activeTagFilters = useMemo(() => Array.from(tagFilters), [tagFilters]);
  const normalizedSearch = searchTerm.trim().toLocaleLowerCase();
  const filteredModels = useMemo(
    () => models.filter((model) =>
      (recommendationFilter === "all" || model.recommendation === recommendationFilter) &&
      activeTagFilters.every((tag) => model.tags.includes(tag)) &&
      (normalizedSearch === "" || `${model.name} ${model.relative_path} ${model.project_name ?? ""} ${model.tags.join(" ")}`.toLocaleLowerCase().includes(normalizedSearch))
    ),
    [activeTagFilters, models, normalizedSearch, recommendationFilter]
  );

  useEffect(() => {
    if (!root || loading) return;
    setTagFilters((current) => {
      if (current.size === 0) {
        return current;
      }
      const next = new Set(Array.from(current).filter((tag) => availableTagSet.has(tag)));
      return next.size === current.size ? current : next;
    });
  }, [availableTagSet, loading, root]);

  useEffect(() => {
    try {
      window.sessionStorage.setItem(MODEL_FILTER_SESSION_KEY, JSON.stringify({
        recommendation: recommendationFilter,
        tags: activeTagFilters,
      }));
    } catch {
      // Filtering remains functional when browser storage is unavailable.
    }
  }, [activeTagFilters, recommendationFilter]);

  useEffect(() => {
    setDraftRecommendation(selectedModel?.recommendation ?? "unrated");
    setDraftTags(selectedModel?.tags ?? []);
    setNewTag("");
  }, [selectedModel?.relative_path]);

  const selectedStatus = selectedModel
    ? selectedModel.artifact_status ?? selectedModel.scan_status
    : artifactStatus(selectedArtifact);
  const selectedKind = selectedModel?.kind ?? selectedArtifact?.kind;
  const selectedIsActive = typeof activeArtifactId === "number"
    && selectedModel?.artifact_id === activeArtifactId;
  const previewBackend = selectedKind === "engine"
    ? "DeepStream 推理"
    : selectedKind === "onnx"
      ? "ONNX（当前主链不可用）"
      : selectedIsActive && runtimeBackend
        ? runtimeBackend
        : "按模型自动选择";
  const previewInputShape = selectedVersion?.input_shape ||
    (selectedIsActive ? runtimeInputShape : "") ||
    "切换时验证输入输出";
  const switchLabel = busy === "model.switch"
    ? "正在验证并切换..."
    : selectedIsActive ? "已是当前部署模型" : "验证并切换到所选模型";
  const metadataDirty = selectedModel !== null && (
    draftRecommendation !== selectedModel.recommendation ||
    draftTags.length !== selectedModel.tags.length ||
    draftTags.some((tag, index) => tag !== selectedModel.tags[index])
  );
  const pendingTag = newTag.trim();
  const pendingTagIsNew = pendingTag !== "" && !draftTags.some((tag) => tag.toLocaleLowerCase() === pendingTag.toLocaleLowerCase());
  const metadataEditable = selectedModel?.kind === "engine" && busy === null;
  const selectionHiddenByFilter = selectedModel !== null && !filteredModels.some((model) => model.relative_path === selectedModel.relative_path);
  const selectedProject = selectedModel?.project_name ?? (selectedModel?.kind === "engine" ? "切换时自动登记" : "不适用");
  const selectedPreparation = selectedModel?.kind !== "engine"
    ? "当前主链不支持"
    : selectedModel?.artifact_status === "ready"
      ? "有验证记录 · 切换时复核"
      : "切换时验证";
  const saveMetadata = () => {
    const tags = pendingTagIsNew ? [...draftTags, pendingTag] : draftTags;
    setDraftTags(tags);
    setNewTag("");
    onSaveMetadata(draftRecommendation, tags);
  };
  const addTag = (value: string) => {
    const tag = value.trim();
    if (!tag || draftTags.some((item) => item.toLocaleLowerCase() === tag.toLocaleLowerCase())) return;
    setDraftTags((current) => [...current, tag]);
    setNewTag("");
  };
  const clearFilters = () => {
    setSearchTerm("");
    setRecommendationFilter("all");
    setTagFilters(new Set());
  };
  return (
    <div className="model-selection-panel">
      <header className="model-selection-toolbar">
        <div>
          <strong>设备模型文件</strong>
          <span>{directoryCount} 个文件夹 · 当前显示 {filteredModels.length}/{modelCount} 个模型；项目与版本在首次使用时自动登记</span>
        </div>
        <button
          className="console-button secondary"
          disabled={busy !== null || metadataDirty || pendingTagIsNew}
          onClick={onRefresh}
          type="button"
        >
          <NovaIcon name="refresh" size={15} />
          {busy === "model.refresh" ? "刷新中..." : "刷新模型"}
        </button>
      </header>

      {catalogMessage ? <div className="model-switch-note good">{catalogMessage}</div> : null}

      <div className={models.length > 0 ? "model-selection-workspace" : "model-selection-workspace empty"}>
        <div className="model-selection-browser">
          {models.length > 0 ? <section className="model-vault-filters" aria-label="模型筛选">
            <label className="model-vault-search">
              <NovaIcon name="search" size={17} />
              <input
                aria-label="查找模型文件"
                onChange={(event) => setSearchTerm(event.target.value)}
                placeholder="搜索文件、路径、项目或标签"
                type="search"
                value={searchTerm}
              />
            </label>
            <div className="model-vault-recommendation-filter" role="group" aria-label="推荐状态筛选">
              {([
                ["all", "全部"],
                ["recommended", "推荐"],
                ["unrated", "待整理"],
                ["not_recommended", "不推荐"]
              ] as const).map(([value, label]) => (
                <button type="button"
                  aria-pressed={recommendationFilter === value}
                  className={recommendationFilter === value ? "active" : ""}
                  key={value}
                  onClick={() => setRecommendationFilter(value)}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="model-vault-tag-filter" aria-label="标签筛选">
              <span>标签筛选</span>
              {availableTags.length > 0 ? availableTags.map((tag) => (
                <button type="button"
                  aria-pressed={tagFilters.has(tag)}
                  className={tagFilters.has(tag) ? "active" : ""}
                  key={tag}
                  onClick={() => setTagFilters((current) => {
                    const next = new Set(current);
                    if (next.has(tag)) next.delete(tag); else next.add(tag);
                    return next;
                  })}
                >
                  {tag}
                </button>
              )) : <small>保存标签后可在这里筛选</small>}
              {tagFilters.size > 0 || recommendationFilter !== "all" || searchTerm ? (
                <button type="button" className="clear" onClick={clearFilters}>清除筛选</button>
              ) : null}
            </div>
          </section> : null}
          {loading && root === null ? (
            <div className="model-catalog-placeholder">正在读取 models 目录...</div>
          ) : filteredModels.length > 0 ? (
            <ModelCatalogTree
              models={filteredModels}
              selectionLocked={metadataDirty || pendingTagIsNew}
              selectedPath={selectedPath}
              activeArtifactId={activeArtifactId}
              onSelectModel={onSelectModel}
            />
          ) : (
            <div className="model-catalog-placeholder">
              {models.length > 0
                ? <><p>没有符合条件的模型文件。</p><button className="console-button" onClick={clearFilters} type="button">清除筛选</button></>
                : "暂无可选模型。将 .engine 文件放入设备的 models 目录后点击“刷新模型”；.onnx 文件不能直接切换到当前主链。"}
            </div>
          )}
        </div>

        {models.length > 0 ? <aside className="model-selection-preview" aria-live="polite">
          <div className="model-selection-preview-heading">
            <span className="model-selection-preview-icon" aria-hidden="true">
              <NovaIcon name="engine" size={20} />
            </span>
            <div>
              <span>所选模型文件</span>
              <strong title={selectedModel?.name ?? selectedArtifact?.path ?? ""}>
                {selectedModel?.name ?? selectedArtifact?.path ?? "尚未选择模型"}
              </strong>
            </div>
            {selectedModel ? <StatusIndicator tone={modelStatusTone(selectedStatus)}>
              {modelStatusLabel(selectedStatus)}
            </StatusIndicator> : null}
          </div>

          <p className="model-selection-next-step">
            {selectionHiddenByFilter
              ? "当前筛选隐藏了已选模型；清除筛选或重新选择后才能切换。"
              : selectedModel === null
              ? "请先从左侧选择模型文件，再查看用途与切换条件。"
              : selectedModel.kind !== "engine"
                ? "这个文件不能直接用于当前主链；请选择 .engine 文件。"
                : selectedIsActive
                  ? activeLoaded ? "此文件已部署且运行中已装载，无需重复切换。" : "此文件已部署，但当前尚未装载；请到运行总览启动主链。"
                  : "此文件尚未部署；只有完成验证并切换后才会改变运行模型。"}
          </p>
          <div className="model-selection-action">
            <span>使用此文件</span>
            <button
              className="console-button primary"
              disabled={busy !== null || !canSwitch || selectedIsActive || selectionHiddenByFilter || metadataDirty || pendingTagIsNew}
              onClick={onSwitch}
              type="button"
            >
              <NovaIcon name="model-switch" size={16} />
              {switchLabel}
            </button>
            {selectedModel?.kind === "engine" ? <small>
              {selectedIsActive ? "当前部署不会重复发布；装载状态请看页面顶部。" : canSwitch ? "运行中切换会先征求确认；失败时保留原因并尝试回滚。" : "当前文件暂不可切换；请检查文件类型和模型状态。"}
            </small> : null}
          </div>
          {selectedModel ? <dl className="model-selection-key-facts">
            <div><dt>所属项目</dt><dd title={selectedProject}>{selectedProject}</dd></div>
            <div><dt>文件大小</dt><dd>{formatModelSize(selectedModel.size_bytes)}</dd></div>
            <div><dt>使用准备</dt><dd>{selectedPreparation}</dd></div>
          </dl> : null}
          <details className="model-selection-advanced">
            <summary>查看文件详情与解析设置</summary>
            <dl className="model-selection-facts">
              <div className="wide">
                <dt>当前部署路径</dt>
                <dd title={activeArtifactPath}>{activeArtifactPath || "未部署模型"}</dd>
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
                <dt>内部版本引用</dt>
                <dd title={selectedVersion?.version ?? selectedModel?.version_name ?? ""}>{selectedVersion?.version === "default" ? "自动发现" : selectedVersion?.version ?? selectedModel?.version_name ?? "首次使用时登记"}</dd>
              </div>
              <div>
                <dt>输入尺寸</dt>
                <dd>{previewInputShape}</dd>
              </div>
              <div>
                <dt>运行方式</dt>
                <dd>{previewBackend}</dd>
              </div>
              <div>
                <dt>切换准备</dt>
                <dd>{selectedArtifact ? modelStatusLabel(selectedArtifact.status) : selectedModel?.kind === "engine" ? "后缀已接受，切换时验证" : "不可加载"}</dd>
              </div>
              <div className="wide model-parser-preset">
                <dt>解析格式</dt>
                <dd>
                  <select
                    aria-label="模型解析格式"
                    disabled={busy !== null}
                    onChange={(event) => onParserPresetChange(event.target.value as ParserPresetId)}
                    value={parserPreset}
                  >
                    <option value="auto">自动识别（推荐）</option>
                    <option value="yolov5">YOLO v5 解析格式</option>
                    <option value="yolov8">YOLO v8 解析格式</option>
                    <option value="yolo11">YOLO v11 解析格式</option>
                    <option value="novasight_generic">NovaSight 通用解析器（内置）</option>
                  </select>
                  <small>目录浏览不加载 Engine；确认切换后才验证真实输入输出。</small>
                </dd>
              </div>
            </dl>
          </details>

          <section className="model-metadata-editor" aria-labelledby="model-metadata-title">
              <div className="model-metadata-heading">
                <div>
                  <strong id="model-metadata-title">整理此模型</strong>
                  <small>{selectedModel?.kind === "engine" ? "推荐状态与标签只用于查找，不会切换或加载模型。" : "当前只支持整理 .engine 文件；.onnx 仅供查看。"}</small>
                </div>
                {metadataDirty ? <span>待保存</span> : null}
              </div>
              <div className="model-recommendation-control" role="group" aria-label="模型推荐状态">
                {RECOMMENDATION_OPTIONS.map((option) => (
                  <button type="button"
                    aria-pressed={draftRecommendation === option.value}
                    className={draftRecommendation === option.value ? "active" : ""}
                    disabled={!metadataEditable}
                    key={option.value}
                    onClick={() => setDraftRecommendation(option.value)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
              <div className="model-tag-editor">
                <div className="model-tag-list">
                  {draftTags.map((tag) => (
                    <button type="button"
                      aria-label={`移除标签 ${tag}`}
                      disabled={!metadataEditable}
                      key={tag}
                      onClick={() => setDraftTags((current) => current.filter((item) => item !== tag))}
                    >
                      {tag}<span aria-hidden="true">×</span>
                    </button>
                  ))}
                  {draftTags.length === 0 ? <small>尚未添加标签</small> : null}
                </div>
                <div className="model-tag-input-row">
                  <input
                    aria-label="新增模型标签"
                    disabled={!metadataEditable}
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
                  <button type="button"
                    className="console-button secondary"
                    disabled={!metadataEditable || !pendingTagIsNew}
                    onClick={() => addTag(newTag)}
                  >
                    添加
                  </button>
                </div>
                <div className="model-tag-suggestions" aria-label="常用标签">
                  {SUGGESTED_MODEL_TAGS.filter((tag) => !draftTags.some((item) => item.toLocaleLowerCase() === tag.toLocaleLowerCase())).map((tag) => (
                    <button type="button" disabled={!metadataEditable} key={tag} onClick={() => addTag(tag)}>
                      + {tag}
                    </button>
                  ))}
                </div>
              </div>
              <div className="console-action-row">
                {(metadataDirty || pendingTagIsNew) ? <button type="button" className="console-button" disabled={busy !== null} onClick={() => {
                  setDraftRecommendation(selectedModel?.recommendation ?? "unrated");
                  setDraftTags(selectedModel?.tags ?? []);
                  setNewTag("");
                }}>放弃整理修改</button> : null}
                <button type="button"
                  className="console-button secondary model-metadata-save"
                  disabled={!metadataEditable || selectionHiddenByFilter || (!metadataDirty && !pendingTagIsNew)}
                  onClick={saveMetadata}
                >
                  {busy === "model.metadata" ? "保存中..." : "保存整理结果"}
                </button>
              </div>
          </section>
        </aside> : null}
      </div>

      {switchMessage ? <div className="model-switch-note good">{switchMessage}</div> : null}
    </div>
  );
}
