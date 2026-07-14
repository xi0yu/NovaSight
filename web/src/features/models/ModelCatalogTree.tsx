import type { ModelCatalogDirectory, ModelCatalogModel } from "../../api";
import { Badge, StatusIndicator } from "../../components/ui";
import { NovaIcon } from "../../components/visual/NovaIcon";

export function ModelCatalogTree({
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
}

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
            onClick={() => onSelectModel(node)}
          >
            <NovaIcon name="models" size={17} />
            <div className="model-catalog-copy">
              <strong>{node.name}</strong>
              <span>
                {node.kind.toUpperCase()} · {formatModelSizeMb(node.size_bytes)}
              </span>
            </div>
            <aside>
              {active ? <Badge tone="good">当前使用</Badge> : null}
              <StatusIndicator tone={statusTone(status)}>
                {node.artifact_id ? statusLabel(status) : "未登记"}
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

function formatModelSizeMb(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "大小未知";
  }
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function statusTone(status: string): "good" | "warn" | "bad" | "idle" {
  if (status === "ready") {
    return "good";
  }
  if (status === "invalid" || status === "failed" || status === "unsupported") {
    return "bad";
  }
  if (status === "need_confirm" || status === "pending" || status === "running") {
    return "warn";
  }
  return "idle";
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    ready: "可用",
    pending: "待验证",
    running: "处理中",
    need_confirm: "待确认",
    invalid: "无效",
    failed: "失败",
    unsupported: "不支持"
  };
  return labels[status] ?? status;
}
