import { useRef, type ChangeEvent, type ReactNode } from "react";

import { NovaIcon } from "../../components/visual";

import type { ConsolePage } from "./StudioNavigation";
import "./settings-view.css";

export function SettingsView({
  modules,
  licenseContent,
  desiredRevision,
  effectiveRevision,
  restartRequired,
  configAvailable,
  runtimeVerified = true,
  operationPending,
  parameterChangesPending,
  onExport,
  onImport,
  onNavigate,
}: {
  modules: readonly string[];
  licenseContent?: ReactNode;
  desiredRevision: number;
  effectiveRevision: number;
  restartRequired: boolean;
  configAvailable: boolean;
  runtimeVerified?: boolean;
  operationPending: boolean;
  parameterChangesPending: boolean;
  onExport: () => void;
  onImport: (file: File) => void | Promise<void>;
  onNavigate: (page: ConsolePage) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const revisionStatus = !configAvailable ? "正在读取设置" : !runtimeVerified ? "正在确认运行状态"
    : restartRequired ? "部分设置重启后生效" : desiredRevision !== effectiveRevision ? "设置正在应用" : "设置已生效";
  const revisionTone = !configAvailable || !runtimeVerified ? "unknown" : revisionStatus === "设置已生效" ? "" : "attention";
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
          <h2>导出配置</h2>
          <p>一键保存当前配置，换机或恢复时直接使用。</p>
        </div>
        <button className="console-button primary" disabled={!configAvailable || operationPending} onClick={onExport} type="button">
          <NovaIcon name="export" size={16} />
          {operationPending ? "正在处理设置" : configAvailable ? "下载备份" : "设置不可用"}
        </button>
      </article>
    ),
    restore: (
      <article className="settings-transfer-row restore">
        <span className="settings-transfer-icon" aria-hidden="true"><NovaIcon name="import" size={20} /></span>
        <div>
          <small>从已有文件恢复</small>
          <h2>导入配置</h2>
          <p>{parameterChangesPending
            ? "算法参数还有未应用的修改。先决定保存或放弃，回来后即可继续恢复。"
            : "选择配置文件，预览改动后确认恢复。不会自动启动设备。"}</p>
        </div>
        {parameterChangesPending ? (
          <button className="console-button attention" onClick={() => onNavigate("params")} type="button">先处理参数</button>
        ) : (
          <button className="console-button" disabled={!configAvailable || operationPending} onClick={() => fileInputRef.current?.click()} type="button">
            <NovaIcon name="import" size={16} />
            {operationPending ? "正在处理设置" : configAvailable ? "选择备份" : "设置不可用"}
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

  };

  return (
    <section className="settings-view" aria-labelledby="settings-transfer-title">
      <header className="settings-hero">
        <div>
          <h2 id="settings-transfer-title">备份与恢复</h2>
          <p>把调好的设置留住，随时安心恢复。</p>
        </div>
        <div className={`settings-revision ${revisionTone}`} role="status">
          <small>当前设置</small>
          <strong>{revisionStatus}</strong>
        </div>
      </header>
      <div className="settings-transfer-list">
        {modules.map((moduleId) => moduleViews[moduleId] ? <div data-module={moduleId} key={moduleId}>{moduleViews[moduleId]}</div> : null)}
      </div>
      {licenseContent ? <div className="settings-license">{licenseContent}</div> : null}
      <footer className="settings-records">
        <div><strong>查看最近操作与问题记录</strong><p>本地最多保留 200 条操作记录，可在实时日志导出。浏览器存储不可用时仅本页保留。</p></div>
        <button className="console-button" type="button" onClick={() => onNavigate("activity")}><NovaIcon name="logs" size={18} />查看问题记录</button>
      </footer>
    </section>
  );
}
