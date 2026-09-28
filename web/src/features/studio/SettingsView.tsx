import { useRef, type ChangeEvent, type ReactNode } from "react";

import { NovaIcon } from "../../components/visual";

import type { ConsolePage } from "./StudioNavigation";
import "./settings-view.css";

export function SettingsView({
  modules,
  desiredRevision,
  effectiveRevision,
  restartRequired,
  configAvailable,
  operationPending,
  parameterChangesPending,
  onExport,
  onImport,
  onNavigate,
}: {
  modules: readonly string[];
  desiredRevision: number;
  effectiveRevision: number;
  restartRequired: boolean;
  configAvailable: boolean;
  operationPending: boolean;
  parameterChangesPending: boolean;
  onExport: () => void;
  onImport: (file: File) => void | Promise<void>;
  onNavigate: (page: ConsolePage) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file && !parameterChangesPending) void onImport(file);
  };
  const moduleViews: Record<string, ReactNode> = {
    backup: (
      <article className="settings-transfer-row backup">
        <span className="settings-transfer-icon" aria-hidden="true"><NovaIcon name="export" size={20} /></span>
        <div>
          <small>保存一份当前状态</small>
          <h2>备份设置</h2>
          <p>下载当前设备已经保存的设置。算法页尚未应用的修改不会混入备份。</p>
        </div>
        <button className="console-button primary" disabled={!configAvailable || operationPending} onClick={onExport} type="button">
          <NovaIcon name="export" size={16} />
          {operationPending ? "正在处理设置" : configAvailable ? "下载备份" : "正在读取设置"}
        </button>
      </article>
    ),
    restore: (
      <article className="settings-transfer-row restore">
        <span className="settings-transfer-icon" aria-hidden="true"><NovaIcon name="import" size={20} /></span>
        <div>
          <small>从已有文件恢复</small>
          <h2>恢复设置</h2>
          <p>{parameterChangesPending
            ? "算法参数还有未应用的修改。先决定保存或放弃，回来后即可继续恢复。"
            : "选择备份后先预览会改变的部分，确认后才会写入设备；首页运行状态不属于配置备份。"}</p>
        </div>
        {parameterChangesPending ? (
          <button className="console-button attention" onClick={() => onNavigate("params")} type="button">先处理参数</button>
        ) : (
          <button className="console-button" disabled={!configAvailable || operationPending} onClick={() => fileInputRef.current?.click()} type="button">
            <NovaIcon name="import" size={16} />
            {operationPending ? "正在处理设置" : configAvailable ? "选择备份" : "正在读取设置"}
          </button>
        )}
        <input
          ref={fileInputRef}
          aria-hidden="true"
          className="visually-hidden"
          tabIndex={-1}
          type="file"
          accept="application/json,.json"
          onChange={onFileChange}
        />
      </article>
    ),
    details: (
      <details className="settings-technical-details">
        <summary>
          <span><b>技术信息</b><small>仅在迁移失败或需要人工核对时查看</small></span>
          <NovaIcon name="forward" size={16} />
        </summary>
        <dl>
          <div><dt>保存版本</dt><dd>{desiredRevision}</dd></div>
          <div><dt>运行版本</dt><dd>{effectiveRevision}</dd></div>
          <div><dt>生效状态</dt><dd>{restartRequired ? "部分设置等待服务重启" : "已与当前运行状态一致"}</dd></div>
          <div><dt>备份格式</dt><dd>NovaSight JSON</dd></div>
        </dl>
      </details>
    ),
  };

  return (
    <section className="settings-view" aria-labelledby="settings-transfer-title">
      <header className="settings-hero">
        <div>
          <h2 id="settings-transfer-title">备份与恢复</h2>
          <p>日常调整留在各自页面。这里仅处理整套设备设置的备份、迁移和恢复。</p>
        </div>
        <div className={restartRequired ? "settings-revision attention" : "settings-revision"} role="status">
          <small>当前设置</small>
          <strong>{restartRequired ? "等待重启" : "版本一致"}</strong>
          <span>{desiredRevision} / {effectiveRevision}</span>
        </div>
      </header>
      <div className="settings-transfer-list">
        {modules.map((moduleId) => moduleViews[moduleId] ? <div data-module={moduleId} key={moduleId}>{moduleViews[moduleId]}</div> : null)}
      </div>
    </section>
  );
}
