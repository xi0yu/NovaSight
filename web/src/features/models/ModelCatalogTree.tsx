import { memo } from "react";

import type { ModelCatalogDirectory, ModelCatalogModel } from "../../api";
import { Badge, StatusIndicator } from "../../components/ui";
import { NovaIcon } from "../../components/visual/NovaIcon";
import { formatModelSize, modelStatusLabel, modelStatusTone } from "./modelPresentation";

export const ModelCatalogTree = memo(function ModelCatalogTree({
  root,
  expandedDirectories,
  selectedPath,
  activeArtifactId,
  onToggleDirectory,
  onSelectModel
}: {
  root: ModelCatalogDirectory;
  expandedDirectories: Set<string>;
  selectedPath: string | undefined;
  activeArtifactId: number | null;
  onToggleDirectory: (relativePath: string) => void;
  onSelectModel: (model: ModelCatalogModel) => void;
}) {
  return (
    <div className="model-catalog" role="tree" aria-label="models 模型目录">
      <CatalogNodes
        nodes={root.children}
        depth={0}
        expandedDirectories={expandedDirectories}
        selectedPath={selectedPath}
        activeArtifactId={activeArtifactId}
        onToggleDirectory={onToggleDirectory}
        onSelectModel={onSelectModel}
      />
    </div>
  );
});

function CatalogNodes({
  nodes,
  depth,
  expandedDirectories,
  selectedPath,
  activeArtifactId,
  onToggleDirectory,
  onSelectModel
}: {
  nodes: Array<ModelCatalogDirectory | ModelCatalogModel>;
  depth: number;
  expandedDirectories: Set<string>;
  selectedPath: string | undefined;
  activeArtifactId: number | null;
  onToggleDirectory: (relativePath: string) => void;
  onSelectModel: (model: ModelCatalogModel) => void;
}) {
  return (
    <div className="model-catalog-level" role="group">
      {nodes.map((node) => {
        if (node.type === "directory") {
          const expanded = expandedDirectories.has(node.relative_path);
          return (
            <div className="model-catalog-branch" key={`directory:${node.relative_path}`}>
              <button
                type="button"
                role="treeitem"
                className="model-catalog-row directory"
                style={{ paddingLeft: `${12 + depth * 18}px` }}
                aria-expanded={expanded}
                onClick={() => onToggleDirectory(node.relative_path)}
              >
                <NovaIcon
                  className={`model-catalog-chevron ${expanded ? "expanded" : ""}`}
                  name="collapse"
                  size={14}
                />
                <NovaIcon name="batch" size={17} />
                <strong>{node.name}</strong>
                <span>{countModels(node)} 个模型</span>
              </button>
              {expanded ? (
                <CatalogNodes
                  nodes={node.children}
                  depth={depth + 1}
                  expandedDirectories={expandedDirectories}
                  selectedPath={selectedPath}
                  activeArtifactId={activeArtifactId}
                  onToggleDirectory={onToggleDirectory}
                  onSelectModel={onSelectModel}
                />
              ) : null}
            </div>
          );
        }

        const selected = node.relative_path === selectedPath;
        const active = node.artifact_id === activeArtifactId;
        const status = node.artifact_status ?? node.scan_status;
        return (
          <button
            type="button"
            role="treeitem"
            key={`model:${node.relative_path}`}
            className={`model-catalog-row model ${selected ? "selected" : ""}`}
            style={{ paddingLeft: `${30 + depth * 18}px` }}
            aria-selected={selected}
            aria-label={`${node.name}，${formatModelSize(node.size_bytes)}，${modelStatusLabel(status)}`}
            onClick={() => onSelectModel(node)}
          >
            <NovaIcon name="models" size={17} />
            <div className="model-catalog-copy">
              <strong title={node.name}>{node.name}</strong>
              <span className="model-catalog-meta">
                <span>{node.kind.toUpperCase()}</span>
                <span className="model-catalog-size">{formatModelSize(node.size_bytes)}</span>
              </span>
            </div>
            <aside>
              {active ? <Badge tone="good">当前使用</Badge> : null}
              <StatusIndicator tone={modelStatusTone(status)}>
                {node.artifact_id ? modelStatusLabel(status) : "未登记"}
              </StatusIndicator>
            </aside>
          </button>
        );
      })}
    </div>
  );
}

function countModels(directory: ModelCatalogDirectory): number {
  return directory.children.reduce(
    (total, child) => total + (child.type === "model" ? 1 : countModels(child)),
    0
  );
}
