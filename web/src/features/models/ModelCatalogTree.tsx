import { memo, useMemo, useState } from "react";

import type {
  ModelCatalogDirectory,
  ModelCatalogModel,
  ModelRecommendation
} from "../../api";
import { Badge, StatusIndicator } from "../../components/ui";
import { NovaIcon } from "../../components/visual/NovaIcon";
import { formatModelSize, modelStatusLabel, modelStatusTone } from "./modelPresentation";

const SHELVES: Array<{
  recommendation: ModelRecommendation;
  label: string;
  description: string;
}> = [
  {
    recommendation: "recommended",
    label: "推荐模型",
    description: "优先用于实际运行"
  },
  {
    recommendation: "unrated",
    label: "待整理",
    description: "尚未给出使用结论"
  },
  {
    recommendation: "not_recommended",
    label: "不推荐模型",
    description: "保留文件，但降低选择优先级"
  }
];

const INITIAL_SHELF_ROWS = 100;

type ShelfModelMap = Record<ModelRecommendation, ModelCatalogModel[]>;

export const ModelCatalogTree = memo(function ModelCatalogTree({
  models,
  selectedPath,
  activeArtifactId,
  onSelectModel
}: {
  models: ModelCatalogModel[];
  selectedPath: string | undefined;
  activeArtifactId: number | null;
  onSelectModel: (model: ModelCatalogModel) => void;
}) {
  const [visibleRows, setVisibleRows] = useState<Record<ModelRecommendation, number>>({
    recommended: INITIAL_SHELF_ROWS,
    unrated: INITIAL_SHELF_ROWS,
    not_recommended: INITIAL_SHELF_ROWS
  });
  const shelfModels = useMemo(() => {
    const grouped: ShelfModelMap = {
      recommended: [],
      unrated: [],
      not_recommended: []
    };
    for (const model of models) {
      grouped[model.recommendation].push(model);
    }
    for (const shelf of SHELVES) {
      grouped[shelf.recommendation].sort((left, right) => {
        const activeOrder = Number(right.artifact_id === activeArtifactId) - Number(left.artifact_id === activeArtifactId);
        return activeOrder || left.name.localeCompare(right.name);
      });
    }
    return grouped;
  }, [activeArtifactId, models]);

  return (
    <div className="model-catalog model-vault-shelves" aria-label="按推荐状态整理的模型" role="list">
      {SHELVES.map((shelf) => {
        const modelsForShelf = shelfModels[shelf.recommendation];
        if (modelsForShelf.length === 0) return null;
        const visibleModels = modelsForShelf.slice(0, visibleRows[shelf.recommendation]);
        return (
          <section
            className={`model-vault-shelf ${shelf.recommendation}`}
            key={shelf.recommendation}
            role="listitem"
          >
            <header className="model-vault-shelf-heading">
              <span aria-hidden="true"><NovaIcon name="batch" size={17} /></span>
              <div>
                <strong>{shelf.label}</strong>
                <small>{shelf.description}</small>
              </div>
              <b>{modelsForShelf.length}</b>
            </header>
            <div className="model-catalog-level" role="group">
              {visibleModels.map((model) => (
                <ModelCatalogRow
                  activeArtifactId={activeArtifactId}
                  key={model.relative_path}
                  model={model}
                  onSelectModel={onSelectModel}
                  selectedPath={selectedPath}
                />
              ))}
              {visibleModels.length < modelsForShelf.length ? (
                <button type="button"
                  className="console-button secondary model-vault-load-more"
                  onClick={() => setVisibleRows((current) => ({
                    ...current,
                    [shelf.recommendation]: current[shelf.recommendation] + INITIAL_SHELF_ROWS
                  }))}
                >
                  再显示 {Math.min(INITIAL_SHELF_ROWS, modelsForShelf.length - visibleModels.length)} 个
                </button>
              ) : null}
            </div>
          </section>
        );
      })}
    </div>
  );
});

function ModelCatalogRow({
  model,
  selectedPath,
  activeArtifactId,
  onSelectModel
}: {
  model: ModelCatalogModel;
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
      aria-label={`${model.name}，${recommendationLabel}，${formatModelSize(model.size_bytes)}，${modelStatusLabel(status)}`}
      aria-pressed={selected}
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
        {active ? <Badge tone="good">当前使用</Badge> : null}
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
