import { useEffect, useMemo, useRef, useState } from "react";

import {
  getConversionJobs,
  getModelArtifacts,
  getModelVersions,
  publishModel,
  rollbackModel,
  type ActiveModel,
  type ModelArtifact,
  type ModelProject
} from "../../api";
import { Badge, EmptyState, InlineError, Panel, StatusIndicator } from "../../components/ui";
import { getErrorMessage } from "../shared/format";
import { Field } from "../shared/Field";

type PublishFeedback = {
  projectId: number;
  versionId: number;
  artifactId: number;
  message?: string;
  error?: string;
};

type RollbackFeedback = {
  projectId: number;
  message?: string;
  error?: string;
};

export function ModelsView({
  projects,
  activeModel,
  error,
  onRuntimeRefresh
}: {
  projects: ModelProject[];
  activeModel: ActiveModel | null;
  error: string | undefined;
  onRuntimeRefresh: () => Promise<void>;
}) {
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<number | null>(null);
  const [versions, setVersions] = useState<
    Awaited<ReturnType<typeof getModelVersions>>
  >([]);
  const [artifacts, setArtifacts] = useState<
    Awaited<ReturnType<typeof getModelArtifacts>>
  >([]);
  const [jobs, setJobs] = useState<Awaited<ReturnType<typeof getConversionJobs>>>([]);
  const [loadingVersions, setLoadingVersions] = useState(false);
  const [loadingArtifacts, setLoadingArtifacts] = useState(false);
  const [loadingJobs, setLoadingJobs] = useState(false);
  const [versionsError, setVersionsError] = useState<string>();
  const [artifactsError, setArtifactsError] = useState<string>();
  const [jobsError, setJobsError] = useState<string>();
  const [publishFeedback, setPublishFeedback] = useState<PublishFeedback | null>(null);
  const [rollbackFeedback, setRollbackFeedback] = useState<RollbackFeedback | null>(null);
  const [publishingArtifactId, setPublishingArtifactId] = useState<number | null>(null);
  const [rollingBack, setRollingBack] = useState(false);
  const [registryRefreshKey, setRegistryRefreshKey] = useState(0);
  const previousProjectId = useRef<number | null>(null);

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId]
  );
  const selectedVersion = useMemo(
    () => versions.find((version) => version.id === selectedVersionId) ?? null,
    [versions, selectedVersionId]
  );
  const visiblePublishFeedback =
    publishFeedback &&
    publishFeedback.projectId === selectedProjectId &&
    publishFeedback.versionId === selectedVersionId
      ? publishFeedback
      : null;
  const visibleRollbackFeedback =
    rollbackFeedback && rollbackFeedback.projectId === selectedProjectId ? rollbackFeedback : null;
  const activeProjectId = activeModel?.project?.id ?? null;
  const readyArtifacts = artifacts.filter((artifact) => artifact.status === "ready");
  const publishableArtifacts = readyArtifacts.filter(
    (artifact) => artifact.kind === "engine" || artifact.kind === "onnx"
  );

  useEffect(() => {
    if (projects.length === 0) {
      setSelectedProjectId(null);
      return;
    }
    setSelectedProjectId((current) =>
      current !== null && projects.some((project) => project.id === current)
        ? current
        : projects[0].id
    );
  }, [projects]);

  useEffect(() => {
    if (selectedProjectId === null) {
      previousProjectId.current = null;
      setVersions([]);
      setSelectedVersionId(null);
      setVersionsError(undefined);
      return;
    }

    const projectChanged = previousProjectId.current !== selectedProjectId;
    previousProjectId.current = selectedProjectId;
    let cancelled = false;
    setLoadingVersions(true);
    setVersionsError(undefined);
    if (projectChanged) {
      setVersions([]);
      setSelectedVersionId(null);
    }

    getModelVersions(selectedProjectId)
      .then((nextVersions) => {
        if (cancelled) {
          return;
        }
        setVersions(nextVersions);
      })
      .catch((requestError) => {
        if (cancelled) {
          return;
        }
        setVersions([]);
        setVersionsError(getErrorMessage(requestError));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setLoadingVersions(false);
      });

    return () => {
      cancelled = true;
    };
  }, [registryRefreshKey, selectedProjectId]);

  useEffect(() => {
    if (versions.length === 0) {
      setSelectedVersionId(null);
      return;
    }
    setSelectedVersionId((current) =>
      current !== null && versions.some((version) => version.id === current)
        ? current
        : versions[0].id
    );
  }, [versions]);

  useEffect(() => {
    if (selectedVersionId === null) {
      setArtifacts([]);
      setJobs([]);
      setArtifactsError(undefined);
      setJobsError(undefined);
      return;
    }

    let cancelled = false;
    setLoadingArtifacts(true);
    setLoadingJobs(true);
    setArtifactsError(undefined);
    setJobsError(undefined);
    setArtifacts([]);
    setJobs([]);

    getModelArtifacts(selectedVersionId)
      .then((nextArtifacts) => {
        if (cancelled) {
          return;
        }
        setArtifacts(nextArtifacts);
      })
      .catch((requestError) => {
        if (cancelled) {
          return;
        }
        setArtifacts([]);
        setArtifactsError(getErrorMessage(requestError));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setLoadingArtifacts(false);
      });

    getConversionJobs(selectedVersionId)
      .then((nextJobs) => {
        if (cancelled) {
          return;
        }
        setJobs(nextJobs);
      })
      .catch((requestError) => {
        if (cancelled) {
          return;
        }
        setJobs([]);
        setJobsError(getErrorMessage(requestError));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setLoadingJobs(false);
      });

    return () => {
      cancelled = true;
    };
  }, [registryRefreshKey, selectedVersionId]);

  async function handlePublish(artifact: ModelArtifact) {
    if (!selectedProject || selectedVersionId === null) {
      return;
    }

    const requestProjectId = selectedProject.id;
    const requestProjectName = selectedProject.name;
    const requestVersionId = selectedVersionId;
    const warning =
      artifact.kind !== "engine" ? "\n\n警告: 这个产物不是 engine，发布后推理可能不可用。" : "";
    const confirmed = window.confirm(
      [
        "确认发布模型？",
        "",
        `项目: ${requestProjectName} (#${requestProjectId})`,
        `产物路径: ${artifact.path}`,
        `产物类型: ${artifact.kind}`,
        `产物状态: ${artifact.status}${warning}`
      ].join("\n")
    );
    if (!confirmed) {
      return;
    }

    setPublishingArtifactId(artifact.id);
    setPublishFeedback(null);

    try {
      const deployment = await publishModel(requestProjectId, artifact.id);
      await onRuntimeRefresh();
      setRegistryRefreshKey((current) => current + 1);
      setPublishFeedback({
        projectId: requestProjectId,
        versionId: requestVersionId,
        artifactId: artifact.id,
        message: `已为 ${requestProjectName} (#${requestProjectId}) 发布 ${artifact.kind} 产物，部署 #${deployment.id}。`
      });
    } catch (requestError) {
      setPublishFeedback({
        projectId: requestProjectId,
        versionId: requestVersionId,
        artifactId: artifact.id,
        error: getErrorMessage(requestError)
      });
    } finally {
      setPublishingArtifactId(null);
    }
  }

  async function handleRollback() {
    if (!selectedProject) {
      return;
    }

    const requestProjectId = selectedProject.id;
    const requestProjectName = selectedProject.name;
    const confirmed = window.confirm(
      `确认回滚 ${requestProjectName} (#${requestProjectId}) 到上一条部署记录？`
    );
    if (!confirmed) {
      return;
    }

    setRollingBack(true);
    setRollbackFeedback(null);

    try {
      const deployment = await rollbackModel(requestProjectId);
      await onRuntimeRefresh();
      setRegistryRefreshKey((current) => current + 1);
      setRollbackFeedback({
        projectId: requestProjectId,
        message: `已回滚 ${requestProjectName} (#${requestProjectId})，当前部署 #${deployment.id} 指向产物 #${deployment.artifact_id}。`
      });
    } catch (requestError) {
      setRollbackFeedback({
        projectId: requestProjectId,
        error: getErrorMessage(requestError)
      });
    } finally {
      setRollingBack(false);
    }
  }

  return (
    <div className="view-grid models-grid registry-grid">
      <Panel title="模型使用" eyebrow="当前运行">
        {activeModel ? (
          <div className="model-usage-card">
            <div>
              <span>正在使用</span>
              <strong>{activeModel.project?.name ?? "未知模型"}</strong>
              <p>{activeModel.artifact?.path ?? "当前部署缺少文件路径"}</p>
            </div>
            <div className="field-grid compact">
              <Field label="部署编号" value={activeModel.deployment.id} mono />
              <Field label="模型类型" value={formatArtifactKind(activeModel.artifact?.kind ?? "")} />
              <Field label="文件状态" value={formatStatusLabel(activeModel.artifact?.status ?? "未知")} />
              <Field label="项目编号" value={activeModel.project?.id ?? "未知"} mono />
            </div>
          </div>
        ) : (
          <EmptyState
            title="还没有选择运行模型"
            detail="选择一个可用模型并发布后，推理链路会绑定到这份模型。"
            command="POST /api/models/projects/{project_id}/publish"
          />
        )}
      </Panel>

      <Panel
        title="模型方案"
        eyebrow="面向使用场景"
        action={<Badge tone={projects.length > 0 ? "good" : "idle"}>{projects.length} 个方案</Badge>}
      >
        <InlineError message={error} />
        {projects.length > 0 ? (
          <div className="model-panel-list">
            {projects.map((project) => (
              <button
                key={project.id}
                type="button"
                className={`selectable-row ${project.id === selectedProjectId ? "selected" : ""}`}
                aria-pressed={project.id === selectedProjectId}
                onClick={() => setSelectedProjectId(project.id)}
              >
                <div>
                  <strong>{project.name}</strong>
                  <span>{project.description || "直接选择这个模型方案用于推理。"}</span>
                </div>
                <aside className="panel-actions">
                  {activeProjectId === project.id ? <Badge tone="good">当前使用</Badge> : null}
                  <code>#{project.id}</code>
                </aside>
              </button>
            ))}
          </div>
        ) : (
          <EmptyState
            title="还没有可选模型"
            detail="后端模型仓库为空。添加模型后，这里会以使用场景展示，而不是展示开发目录。"
          />
        )}
      </Panel>

      <Panel
        title="部署保护"
        eyebrow={selectedProject ? `${selectedProject.name} · #${selectedProject.id}` : "等待项目"}
      >
        <InlineError message={visibleRollbackFeedback?.error} />
        {visibleRollbackFeedback?.message ? (
          <div className="action-message">{visibleRollbackFeedback.message}</div>
        ) : null}
        {selectedProject ? (
          <>
            <div className="field-grid compact">
              <Field label="项目" value={selectedProject.name} />
              <Field label="项目编号" value={selectedProject.id} mono />
            </div>
            <div className="panel-actions">
              <button
                className="button compact-button"
                type="button"
                onClick={handleRollback}
                disabled={rollingBack || publishingArtifactId !== null}
              >
                {rollingBack ? "回滚中..." : "回到上一模型"}
              </button>
            </div>
          </>
        ) : (
          <EmptyState
            title="没有选中的项目"
            detail="先在模型项目列表中选择一个项目，再执行部署回滚。"
          />
        )}
      </Panel>

      <Panel
        title="模型版本"
        eyebrow={selectedProject ? `${selectedProject.name} · #${selectedProject.id}` : "等待项目"}
      >
        <InlineError message={versionsError} />
        {loadingVersions ? (
          <EmptyState title="正在加载版本" detail="读取项目版本、源模型与输入规格。" />
        ) : versions.length > 0 ? (
          <div className="model-panel-list">
            {versions.map((version) => (
              <button
                key={version.id}
                type="button"
                className={`selectable-row ${version.id === selectedVersionId ? "selected" : ""}`}
                aria-pressed={version.id === selectedVersionId}
                onClick={() => setSelectedVersionId(version.id)}
              >
                <div>
                  <strong>{version.version}</strong>
                  <span>
                    {formatArtifactKind(version.source_kind)} · {version.input_shape || "未标注输入规格"}
                  </span>
                  <span className="mono">{version.source_path}</span>
                </div>
                <aside className="panel-actions">
                  <Badge tone="idle">{version.classes.length} 类</Badge>
                  <code>#{version.id}</code>
                </aside>
              </button>
            ))}
          </div>
        ) : (
          <EmptyState
            title={selectedProject ? "这个项目还没有版本" : "没有选中的项目"}
            detail={
              selectedProject
                ? "后端返回的版本列表为空。"
                : "选择左侧模型项目后，这里会显示版本、源路径和输入规格。"
            }
          />
        )}
      </Panel>

      <Panel
        title="可用模型文件"
        eyebrow={selectedVersion ? `${selectedVersion.version} · #${selectedVersion.id}` : "等待版本"}
        action={
          selectedVersion ? (
            <Badge tone={publishableArtifacts.length > 0 ? "good" : "idle"}>
              {publishableArtifacts.length} 个可发布
            </Badge>
          ) : null
        }
      >
        <InlineError message={artifactsError ?? visiblePublishFeedback?.error} />
        {visiblePublishFeedback?.message ? (
          <div className="action-message">{visiblePublishFeedback.message}</div>
        ) : null}
        {loadingArtifacts ? (
          <EmptyState title="正在加载产物" detail="读取转换结果、校验摘要和发布状态。" />
        ) : artifacts.length > 0 ? (
          <div className="model-panel-list">
            {artifacts.map((artifact) => (
              <article className="project-row" key={artifact.id}>
                <div>
                  <strong>{formatArtifactKind(artifact.kind)}</strong>
                  <span className="mono">{artifact.path}</span>
                  <span className="mono">{artifact.checksum}</span>
                </div>
                <aside className="panel-actions">
                  <StatusIndicator tone={getStatusTone(artifact.status)}>
                    {formatStatusLabel(artifact.status)}
                  </StatusIndicator>
                  <button
                    className="button compact-button"
                    type="button"
                    onClick={() => handlePublish(artifact)}
                    disabled={rollingBack || publishingArtifactId !== null}
                  >
                    {publishingArtifactId === artifact.id ? "应用中..." : "设为当前"}
                  </button>
                </aside>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            title={selectedVersion ? "这个版本还没有产物" : "没有选中的版本"}
            detail={
              selectedVersion
                ? "后端返回的模型文件为空。"
                : "选择一个版本后，这里会显示可直接用于推理的模型文件。"
            }
          />
        )}
      </Panel>

      <Panel
        title="高级转换记录"
        eyebrow={selectedVersion ? `${selectedVersion.version} · #${selectedVersion.id}` : "等待版本"}
      >
        <InlineError message={jobsError} />
        {loadingJobs ? (
          <EmptyState title="正在加载任务" detail="读取转换目标、命令行与日志摘要。" />
        ) : jobs.length > 0 ? (
          <div className="model-panel-list">
            {jobs.map((job) => (
              <article className="project-row" key={job.id}>
                <div>
                  <strong>{formatArtifactKind(job.target_kind)}</strong>
                  <span className="mono">{formatCommand(job.command)}</span>
                  <span>{formatLogPreview(job.log)}</span>
                </div>
                <aside className="panel-actions">
                  <StatusIndicator tone={getStatusTone(job.status)}>
                    {formatStatusLabel(job.status)}
                  </StatusIndicator>
                  <code>#{job.id}</code>
                </aside>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            title={selectedVersion ? "这个版本还没有转换任务" : "没有选中的版本"}
            detail={
              selectedVersion
                ? "后端返回的转换任务为空。"
                : "选择一个版本后，这里会显示转换命令与日志预览。"
            }
          />
        )}
      </Panel>
    </div>
  );
}

function getStatusTone(status: string): "good" | "warn" | "bad" | "idle" {
  if (status === "ready" || status === "succeeded") {
    return "good";
  }
  if (status === "failed") {
    return "bad";
  }
  if (status === "running") {
    return "warn";
  }
  return "idle";
}

function formatArtifactKind(kind: string): string {
  if (kind === "engine") {
    return "TensorRT 引擎";
  }
  if (kind === "onnx") {
    return "ONNX 模型";
  }
  if (kind === "pt") {
    return "训练权重";
  }
  return kind || "未知类型";
}

function formatStatusLabel(status: string): string {
  if (status === "ready" || status === "succeeded") {
    return "可用";
  }
  if (status === "failed") {
    return "失败";
  }
  if (status === "running") {
    return "处理中";
  }
  if (status === "pending") {
    return "等待中";
  }
  return status || "未知";
}

function formatCommand(command: string[]): string {
  if (command.length === 0) {
    return "没有命令参数";
  }
  return command.join(" ");
}

function formatLogPreview(log: string): string {
  const normalized = log.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "无日志";
  }
  return normalized.length > 180 ? `${normalized.slice(0, 177)}...` : normalized;
}
