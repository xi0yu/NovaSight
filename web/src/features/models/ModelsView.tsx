import { type ActiveModel, type ModelProject } from "../../api";
import { EmptyState, InlineError, Panel } from "../../components/ui";
import { Field } from "../shared/Field";

export function ModelsView({
  projects,
  activeModel,
  error
}: {
  projects: ModelProject[];
  activeModel: ActiveModel | null;
  error: string | undefined;
}) {
  return (
    <div className="view-grid models-grid">
      <Panel title="当前模型" eyebrow="运行绑定">
        {activeModel ? (
          <div className="field-grid">
            <Field label="项目" value={activeModel.project?.name ?? "未知项目"} />
            <Field label="部署编号" value={activeModel.deployment.id} mono />
            <Field label="模型文件" value={activeModel.artifact?.path ?? "缺少文件"} mono />
            <Field label="文件状态" value={activeModel.artifact?.status ?? "未知"} />
          </div>
        ) : (
          <EmptyState
            title="没有发布模型"
            detail="发布可用模型后，推理运行时会在这里显示绑定状态。"
            command="POST /api/models/projects/{project_id}/publish"
          />
        )}
      </Panel>
      <Panel title="模型项目" eyebrow="注册表">
        <InlineError message={error} />
        {projects.length > 0 ? (
          <div className="project-list">
            {projects.map((project) => (
              <article className="project-row" key={project.id}>
                <div>
                  <strong>{project.name}</strong>
                  <span>{project.description || "没有描述"}</span>
                </div>
                <code>#{project.id}</code>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            title="模型注册表为空"
            detail="创建模型项目、添加版本、转换产物并发布后，这里会展示项目。"
          />
        )}
      </Panel>
    </div>
  );
}
