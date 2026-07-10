import { useEffect, useMemo, useRef, useState } from "react";

import {
  getConversionJobs,
  getModelArtifacts,
  getModelVersions,
  prepareYolov8nExample,
  publishModel,
  rollbackModel,
  scanModelDirectory,
  uploadModelFile,
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

type InferenceFeedback = {
  loaded: boolean;
  available: boolean;
  selected: string;
  reason: string;
  inputShape: string;
  outputShape: string;
};

function readInferenceFeedback(value: Record<string, unknown> | undefined): InferenceFeedback | null {
  if (!value) {
    return null;
  }
  return {
    loaded: value.loaded === true,
    available: value.available === true,
    selected: typeof value.selected === "string" ? value.selected : "",
    reason: typeof value.reason === "string" ? value.reason : "",
    inputShape: typeof value.input_shape === "string" ? value.input_shape : "",
    outputShape: typeof value.output_shape === "string" ? value.output_shape : ""
  };
}

function isInferenceBindingReady(status: InferenceFeedback | null): boolean {
  if (!status) {
    return false;
  }
  if (status.loaded && status.available) {
    return true;
  }
  return false;
}

function inferenceBindingMessage(status: InferenceFeedback | null, modelName: string, artifactKind: string): string {
  if (!status) {
    return `已发布 ${artifactKind} 产物，请刷新运行状态确认推理绑定。`;
  }
  if (status.loaded) {
    return `已为 ${modelName} 绑定 ${artifactKind}，推理运行时已加载。`;
  }
  return `已发布 ${artifactKind} 产物，但推理运行时未加载成功，请查看下方原因。`;
}

export function ModelsView({
  projects,
  activeModel,
  runtimeInference,
  error,
  onRuntimeRefresh,
  onOpenInference
}: {
  projects: ModelProject[];
  activeModel: ActiveModel | null;
  runtimeInference?: Record<string, unknown>;
  error: string | undefined;
  onRuntimeRefresh: () => Promise<void>;
  onOpenInference: () => void;
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
  const [preparingExample, setPreparingExample] = useState(false);
  const [uploadingModel, setUploadingModel] = useState(false);
  const [scanningModels, setScanningModels] = useState(false);
  const [modelActionMessage, setModelActionMessage] = useState<string>();
  const [modelActionError, setModelActionError] = useState<string>();
  const [inferenceFeedback, setInferenceFeedback] = useState<InferenceFeedback | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadProjectName, setUploadProjectName] = useState("custom_model");
  const [uploadVersion, setUploadVersion] = useState("v1");
  const [uploadClasses, setUploadClasses] = useState("target");
  const [uploadInputShape, setUploadInputShape] = useState("");
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
  const publishableArtifacts = artifacts.filter(
    (artifact) =>
      (artifact.status === "ready" ||
        (artifact.kind === "engine" && artifact.status === "pending")) &&
      (artifact.kind === "engine" || artifact.kind === "onnx")
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
    const confirmed = window.confirm(
      [
        "确认发布模型？",
        "",
        `项目: ${requestProjectName} (#${requestProjectId})`,
        `产物路径: ${artifact.path}`,
        `产物类型: ${artifact.kind}`,
        `产物状态: ${artifact.status}`
      ].join("\n")
    );
    if (!confirmed) {
      return;
    }

    setPublishingArtifactId(artifact.id);
    setPublishFeedback(null);

    try {
      const result = await publishModel(requestProjectId, artifact.id);
      const nextInferenceFeedback = readInferenceFeedback(result.inference);
      setInferenceFeedback(nextInferenceFeedback);
      await onRuntimeRefresh();
      setRegistryRefreshKey((current) => current + 1);
      setPublishFeedback({
        projectId: requestProjectId,
        versionId: requestVersionId,
        artifactId: artifact.id,
        message: inferenceBindingMessage(
          nextInferenceFeedback,
          `${requestProjectName} (#${requestProjectId})`,
          artifact.kind
        )
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
      const result = await rollbackModel(requestProjectId);
      const nextInferenceFeedback = readInferenceFeedback(result.inference);
      setInferenceFeedback(nextInferenceFeedback);
      await onRuntimeRefresh();
      setRegistryRefreshKey((current) => current + 1);
      setRollbackFeedback({
        projectId: requestProjectId,
        message: isInferenceBindingReady(nextInferenceFeedback)
          ? `已回滚 ${requestProjectName} (#${requestProjectId})，推理绑定已更新。`
          : `已回滚 ${requestProjectName} (#${requestProjectId})，但推理运行时未加载成功。`
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

  async function handlePrepareExample() {
    setPreparingExample(true);
    setModelActionError(undefined);
    setModelActionMessage(undefined);
    try {
      const result = await prepareYolov8nExample();
      await onRuntimeRefresh();
      setRegistryRefreshKey((current) => current + 1);
      setSelectedProjectId(result.project.id);
      setSelectedVersionId(result.version.id);
      setModelActionMessage(
        result.downloaded
          ? "YOLOv8n 测试模型已下载并注册。"
          : "YOLOv8n 测试模型已存在，已复用本地文件。"
      );
    } catch (requestError) {
      setModelActionError(getErrorMessage(requestError));
    } finally {
      setPreparingExample(false);
    }
  }

  async function handleScanModels() {
    setScanningModels(true);
    setModelActionError(undefined);
    setModelActionMessage(undefined);
    try {
      const result = await scanModelDirectory();
      await onRuntimeRefresh();
      setRegistryRefreshKey((current) => current + 1);
      setModelActionMessage(
        `已扫描服务端 models 目录，当前发现 ${result.project_count} 个模型方案。`
      );
    } catch (requestError) {
      setModelActionError(getErrorMessage(requestError));
    } finally {
      setScanningModels(false);
    }
  }

  async function handleUploadModel() {
    if (!uploadFile) {
      setModelActionError("请选择 .pt、.onnx 或 .engine 模型文件。");
      return;
    }
    setUploadingModel(true);
    setModelActionError(undefined);
    setModelActionMessage(undefined);
    try {
      const result = await uploadModelFile({
        projectName: uploadProjectName.trim() || "custom_model",
        version: uploadVersion.trim() || "v1",
        description: "前端上传模型",
        classes: uploadClasses,
        inputShape: uploadInputShape,
        file: uploadFile
      });
      await onRuntimeRefresh();
      setRegistryRefreshKey((current) => current + 1);
      setSelectedProjectId(result.project.id);
      setSelectedVersionId(result.version.id);
      setModelActionMessage(`已上传 ${uploadFile.name}，可在可用模型文件中设为当前。`);
      setUploadFile(null);
    } catch (requestError) {
      setModelActionError(getErrorMessage(requestError));
    } finally {
      setUploadingModel(false);
    }
  }

  return (
    <div className="view-grid models-grid registry-grid">
      <Panel title="模型使用台" eyebrow="准备、上传、选择">
        <InlineError message={modelActionError} />
        {modelActionMessage ? <div className="action-message">{modelActionMessage}</div> : null}
        <div className="model-workbench">
          <article className="model-workbench-card">
            <div>
              <strong>服务端 models 目录</strong>
              <p>把 .onnx / .engine 放进服务端 models 目录后，点击扫描即可同步到模型仓库。</p>
            </div>
            <button
              className="button compact-button"
              type="button"
              onClick={handleScanModels}
              disabled={scanningModels || preparingExample || uploadingModel}
            >
              {scanningModels ? "扫描中..." : "扫描 models 目录"}
            </button>
          </article>

          <article className="model-workbench-card">
            <div>
              <strong>测试模型</strong>
              <p>自动准备开源 YOLOv8n，用图片输入源验证推理链路。</p>
            </div>
            <button
              className="button compact-button"
              type="button"
              onClick={handlePrepareExample}
              disabled={preparingExample || uploadingModel || scanningModels}
            >
              {preparingExample ? "准备中..." : "准备 YOLOv8n"}
            </button>
          </article>

          <article className="model-workbench-card upload">
            <div>
              <strong>上传模型</strong>
              <p>上传 .pt、.onnx 或 .engine 文件，注册为可使用模型。</p>
            </div>
            <div className="model-upload-grid">
              <label>
                <span>模型名称</span>
                <input value={uploadProjectName} onChange={(event) => setUploadProjectName(event.target.value)} />
              </label>
              <label>
                <span>版本</span>
                <input value={uploadVersion} onChange={(event) => setUploadVersion(event.target.value)} />
              </label>
              <label>
                <span>类别</span>
                <input value={uploadClasses} onChange={(event) => setUploadClasses(event.target.value)} />
              </label>
              <label>
                <span>输入尺寸</span>
                <input
                  placeholder="自动识别，或填写 1x3x256x256 / 256x256"
                  value={uploadInputShape}
                  onChange={(event) => setUploadInputShape(event.target.value)}
                />
              </label>
              <label className="model-file-input">
                <span>模型文件</span>
                <input
                  accept=".pt,.onnx,.engine"
                  type="file"
                  onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)}
                />
              </label>
              <button
                className="button compact-button"
                type="button"
                onClick={handleUploadModel}
                disabled={uploadingModel || preparingExample || scanningModels}
              >
                {uploadingModel ? "上传中..." : "上传并注册"}
              </button>
            </div>
          </article>
        </div>
      </Panel>

      <Panel title="模型使用" eyebrow="当前运行">
        {activeModel ? (
          <div className="model-usage-card">
            <div>
              <span>正在使用</span>
              <strong>{activeModel.project?.name ?? "未知模型"}</strong>
              <p>{activeModel.artifact?.path ?? "当前部署缺少文件路径"}</p>
            </div>
            <InferenceBindingCard feedback={inferenceFeedback} runtimeInference={runtimeInference} />
            <div className="field-grid compact">
              <Field label="部署编号" value={activeModel.deployment.id} mono />
              <Field label="模型类型" value={formatArtifactKind(activeModel.artifact?.kind ?? "")} />
              <Field label="文件状态" value={formatStatusLabel(activeModel.artifact?.status ?? "未知")} />
              <Field label="项目编号" value={activeModel.project?.id ?? "未知"} mono />
            </div>
            <div className="panel-actions">
              <button className="button compact-button" type="button" onClick={onOpenInference}>
                去推理设置
              </button>
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
                  <strong>{version.version === "default" ? "自动发现版本" : version.version}</strong>
                  <span>
                    {formatArtifactKind(version.source_kind)} · 输入 {version.input_shape || "未标注"}
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
                  <span>{artifactRuntimeHint(artifact)}</span>
                  <span>{formatModelSizeMb(artifact.size_bytes)}</span>
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
                    disabled={
                      rollingBack ||
                      publishingArtifactId !== null ||
                      !(
                        artifact.status === "ready" ||
                        (artifact.kind === "engine" && artifact.status === "pending")
                      ) ||
                      !isRunnableArtifact(artifact)
                    }
                    title={
                      isRunnableArtifact(artifact)
                        ? "根据后缀自动选择 ONNXRuntime 或 TensorRT"
                        : "训练权重不能直接推理，需要先导出 ONNX 或 Engine"
                    }
                  >
                    {publishingArtifactId === artifact.id
                      ? "绑定中..."
                      : artifact.status === "pending"
                        ? "验证并用于推理"
                        : "用于推理"}
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

function isRunnableArtifact(artifact: ModelArtifact): boolean {
  return artifact.kind === "onnx" || artifact.kind === "engine";
}

function artifactRuntimeHint(artifact: ModelArtifact): string {
  if (artifact.kind === "engine") {
    return artifact.status === "pending"
      ? "TensorRT Engine 尚未验证；发布时会先安全加载，失败不会替换当前模型。"
      : "TensorRT Engine，发布后由自定义 TensorRT 主链加载。";
  }
  if (artifact.kind === "onnx") {
    return "后缀 .onnx，发布后自动使用 ONNXRuntime。";
  }
  if (artifact.kind === "pt") {
    return "训练权重不能直接推理，需要先导出 ONNX 或 TensorRT Engine。";
  }
  return "未知文件类型，不能确认推理后端。";
}

function InferenceBindingCard({
  feedback,
  runtimeInference
}: {
  feedback: InferenceFeedback | null;
  runtimeInference?: Record<string, unknown>;
}) {
  const status = feedback ?? readInferenceFeedback(runtimeInference);
  if (!status) {
    return (
      <div className="model-inference-binding idle">
        <strong>推理绑定状态</strong>
        <span>等待运行时状态。选择 ONNX 或 Engine 后，这里会显示加载结果。</span>
      </div>
    );
  }
  const ok = isInferenceBindingReady(status);
  return (
    <div className={ok ? "model-inference-binding good" : "model-inference-binding bad"}>
      <div>
        <strong>
          {status.loaded
            ? "推理运行时已加载"
            : ok
            ? "自定义推理已准备"
            : "推理运行时未就绪"}
        </strong>
        <span>
          {status.selected || "未知后端"}
          {status.inputShape ? ` · 输入 ${status.inputShape}` : ""}
          {status.outputShape ? ` · 输出 ${status.outputShape}` : ""}
        </span>
      </div>
      {status.reason ? <p>{status.reason}</p> : null}
    </div>
  );
}

function formatModelSizeMb(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return "大小不可用";
  }
  return `${(value / 1_000_000).toFixed(2)} MB`;
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
