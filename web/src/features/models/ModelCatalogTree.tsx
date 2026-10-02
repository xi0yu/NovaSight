import { memo, useMemo, useState } from "react";

import type {
  ModelCatalogDirectory,
  ModelCatalogModel
} from "../../api";
import { Badge, StatusIndicator } from "../../components/ui";
import { NovaIcon } from "../../components/visual/NovaIcon";
import { formatModelSize, modelStatusLabel, modelStatusTone } from "./modelPresentation";

const INITIAL_ROWS = 24;
const nameCollator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

export type ModelSortOrder = "name_asc" | "name_desc" | "size_desc" | "size_asc" | "active_first";

export function findCatalogDirectory(root: ModelCatalogDirectory | null, path: string): ModelCatalogDirectory | null {
  if (!root) return null;
  if (!path) return root;
  let directory = root;
  for (const part of path.split("/")) {
    const child = directory.children.find((node) => node.type === "directory" && node.name === part);
    if (!child || child.type !== "directory") return null;
    directory = child;
  }
  return directory;
}

export function sortCatalogModels(models: ModelCatalogModel[], order: ModelSortOrder, activeArtifactId: number | null): ModelCatalogModel[] {
  const byName = (left: ModelCatalogModel, right: ModelCatalogModel) =>
    nameCollator.compare(left.name, right.name) || nameCollator.compare(left.relative_path, right.relative_path);
  return [...models].sort((left, right) => {
    if (order === "active_first") {
      const activeOrder = Number(right.artifact_id === activeArtifactId) - Number(left.artifact_id === activeArtifactId);
      return activeOrder || byName(left, right);
    }
    if (order === "name_desc") return -byName(left, right);
    if (order === "size_desc") return right.size_bytes - left.size_bytes || byName(left, right);
    if (order === "size_asc") return left.size_bytes - right.size_bytes || byName(left, right);
    return byName(left, right);
  });
}

export const ModelCatalogTree = memo(function ModelCatalogTree({
  folders,
  models,
  sortOrder,
  selectionLocked = false,
  selectedPath,
  activeArtifactId,
  onOpenFolder,
  onSelectModel
}: {
  folders: ModelCatalogDirectory[];
  models: ModelCatalogModel[];
  sortOrder: ModelSortOrder;
  selectionLocked?: boolean;
  selectedPath: string | undefined;
  activeArtifactId: number | null;
  onOpenFolder: (path: string) => void;
  onSelectModel: (model: ModelCatalogModel) => void;
}) {
  const [visibleRows, setVisibleRows] = useState(INITIAL_ROWS);
  const sortedModels = useMemo(() => sortCatalogModels(models, sortOrder, activeArtifactId), [activeArtifactId, models, sortOrder]);
  const visibleModels = sortedModels.slice(0, visibleRows);
  const sortedFolders = useMemo(() => [...folders].sort((left, right) => nameCollator.compare(left.name, right.name)), [folders]);

  return (
    <div className="model-catalog model-browser-list" aria-label="模型文件与文件夹" role="list">
      {sortedFolders.map((folder) => (
        <div key={folder.relative_path} role="listitem">
          <button
            aria-label={`打开文件夹 ${folder.name}，包含 ${flattenCatalogModels(folder).length} 个模型`}
            className="model-catalog-row folder"
            disabled={selectionLocked}
            onClick={() => onOpenFolder(folder.relative_path)}
            type="button"
          >
            <span className="model-folder-icon" aria-hidden="true"><NovaIcon name="batch" size={17} /></span>
            <span className="model-folder-copy"><strong>{folder.name}</strong><small>{flattenCatalogModels(folder).length} 个模型</small></span>
            <NovaIcon name="forward" size={16} aria-hidden="true" />
          </button>
        </div>
      ))}
      {visibleModels.map((model) => (
        <div key={model.relative_path} role="listitem">
          <ModelCatalogRow
            activeArtifactId={activeArtifactId}
            disabled={selectionLocked}
            model={model}
            onSelectModel={onSelectModel}
            selectedPath={selectedPath}
          />
        </div>
      ))}
      {visibleModels.length < sortedModels.length ? (
        <button type="button"
          className="console-button secondary model-vault-load-more"
          aria-label={`再显示 ${Math.min(INITIAL_ROWS, sortedModels.length - visibleModels.length)} 个模型`}
          onClick={() => setVisibleRows((current) => current + INITIAL_ROWS)}
        >
          再显示 {Math.min(INITIAL_ROWS, sortedModels.length - visibleModels.length)} 个模型（已显示 {visibleModels.length}/{sortedModels.length}）
        </button>
      ) : null}
    </div>
  );
});

function ModelCatalogRow({
  model,
  disabled,
  selectedPath,
  activeArtifactId,
  onSelectModel
}: {
  model: ModelCatalogModel;
  disabled: boolean;
  selectedPath: string | undefined;
  activeArtifactId: number | null;
  onSelectModel: (model: ModelCatalogModel) => void;
}) {
  const selected = model.relative_path === selectedPath;
  const active = model.artifact_id === activeArtifactId;
  const status = model.artifact_status ?? model.scan_status;
  const recommendationLabel = model.recommendation === "recommended"
    ? "推荐"
    : model.recommendation === "not_recommended"
      ? "不推荐"
      : "待整理";
  return (
    <button type="button"
      aria-label={`${model.name}，路径 ${model.relative_path}，${active ? "已部署，" : ""}${recommendationLabel}，${formatModelSize(model.size_bytes)}，${modelStatusLabel(status)}`}
      aria-pressed={selected}
      disabled={disabled}
      className={`model-catalog-row model ${selected ? "selected" : ""}`}
      data-active={active ? "true" : undefined}
      onClick={() => onSelectModel(model)}
    >
      <NovaIcon name="models" size={17} />
      <div className="model-catalog-copy">
        <strong title={model.name}>{model.name}</strong>
        <span className="model-catalog-meta">
          <span>{model.kind.toUpperCase()}</span>
          <span className="model-catalog-size">{formatModelSize(model.size_bytes)}</span>
          {model.project_name ? <span className="model-catalog-project" title={`所属项目：${model.project_name}`}>项目：{model.project_name}</span> : null}
          <span className="model-catalog-path" title={model.relative_path}>{model.relative_path}</span>
        </span>
        {model.tags.length > 0 ? (
          <span className="model-catalog-tags" aria-label={`标签：${model.tags.join("、")}`}>
            {model.tags.slice(0, 3).map((tag) => <i key={tag}>{tag}</i>)}
            {model.tags.length > 3 ? <i>+{model.tags.length - 3}</i> : null}
          </span>
        ) : null}
      </div>
      <aside>
        {active ? <Badge tone="good">已部署</Badge> : null}
        {model.recommendation === "recommended" ? <Badge tone="good">推荐</Badge> : null}
        {model.recommendation === "not_recommended" ? <Badge tone="warn">不推荐</Badge> : null}
        <StatusIndicator tone={modelStatusTone(status)}>
          {model.artifact_id ? modelStatusLabel(status) : "未登记"}
        </StatusIndicator>
      </aside>
    </button>
  );
}

export function flattenCatalogModels(root: ModelCatalogDirectory | null): ModelCatalogModel[] {
  if (!root) return [];
  return root.children.flatMap((node) =>
    node.type === "model" ? [node] : flattenCatalogModels(node)
  );
}
