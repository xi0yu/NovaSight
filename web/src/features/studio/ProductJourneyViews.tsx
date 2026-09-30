import type { LicenseStatus, RuntimeState } from "../../api";
import { NovaIcon, type NovaIconName } from "../../components/visual";
import { version as studioVersion } from "../../../package.json";
import type { RuntimeProjection } from "../runtime/runtimeProjection";
import type { ConsolePage } from "./StudioNavigation";
import type { ReactNode } from "react";
import companionArt from "../../assets/themes/nova-companion.webp";

import "./product-journey.css";

type Navigate = (page: ConsolePage) => void;

export type SetupState = {
  captureReady: boolean;
  modelReady: boolean;
  configReady: boolean;
  runtimeReady: boolean;
};

const SETUP_STEP_COUNT = 4;

function completedSetupSteps(state: SetupState): number {
  return [state.captureReady, state.modelReady, state.configReady, state.runtimeReady]
    .filter(Boolean).length;
}

export function HomeWelcome({ onNavigate, children }: { onNavigate: Navigate; children: ReactNode }) {
  return <section className="home-room" aria-labelledby="home-welcome-title">
    <header className="home-room-heading">
      <div><span className="studio-kicker">NOVA / DAYLIGHT STUDIO</span><h2 id="home-welcome-title">今天，也请多关照。</h2></div>
      <span className="home-theme-stamp" aria-label="星野主题">✦ 星野工作室</span>
    </header>
    <div className="home-room-layout">
      <aside className="home-companion" aria-label="工作室引导">
        <div className="companion-orbit" aria-hidden="true" />
        <span className="companion-sticker" aria-hidden="true">HELLO, NOVA! <i>✧</i></span>
        <img className="companion-portrait" src={companionArt} width="1024" height="1536" alt="" aria-hidden="true" />
        <div className="companion-dialogue">
          <span className="companion-name">NOVA <small>工作室向导</small></span>
          <p>先调成喜欢的手感，<br />再由你决定什么时候开始。</p>
          <button type="button" onClick={() => onNavigate("onboarding")}>带我完成设置 <span aria-hidden="true">↗</span></button>
        </div>
      </aside>
      <div className="home-control-desk">{children}</div>
    </div>
  </section>;
}

export function HomeSetupPrompt({ state, statusKnown, onNavigate }: { state: SetupState; statusKnown: boolean; onNavigate: Navigate }) {
  const completed = completedSetupSteps(state);
  if (!statusKnown || (state.captureReady && state.modelReady && state.configReady)) return null;
  const next = !state.captureReady
    ? { page: "capture" as const, label: "选择画面设备", detail: "连接采集设备，选择画面来源与画质。" }
    : !state.modelReady
      ? { page: "models" as const, label: "选择识别模型", detail: "为当前设备准备一个可用的识别模型。" }
      : { page: "params" as const, label: "确认使用参数", detail: "设置辅助范围、目标规则与跟随手感。" };
  return (
    <section className="home-setup-prompt" aria-labelledby="home-setup-title">
      <span className="home-setup-icon" aria-hidden="true"><NovaIcon name="play-circle" size={20} /></span>
      <div>
        <small>首次设置 · {completed}/{SETUP_STEP_COUNT}</small>
        <h2 id="home-setup-title">{next.label}</h2>
        <p>{next.detail}</p>
      </div>
      <button className="console-button primary" onClick={() => onNavigate(next.page)} type="button">
        {next.label}
      </button>
    </section>
  );
}

export function HomeShortcuts({ onNavigate }: { onNavigate: Navigate }) {
  return <nav className="home-shortcuts" aria-label="常用操作">
    {([
      { page: "capture", icon: "capture", label: "设备与画面", detail: "选择设备 · 调整画质" },
      { page: "params", icon: "target", label: "调整使用手感", detail: "范围 · 瞄点 · 跟随" },
      { page: "activity", icon: "logs", label: "查看使用记录", detail: "运行日志 · 问题排查" },
    ] as const).map((item) => <button type="button" key={item.page} onClick={() => onNavigate(item.page)}>
      <NovaIcon name={item.icon} size={20} /><span><b>{item.label}</b><small>{item.detail}</small></span><i aria-hidden="true">→</i>
    </button>)}
  </nav>;
}

export function OnboardingView({ state, statusKnown, onNavigate }: { state: SetupState; statusKnown: boolean; onNavigate: Navigate }) {
  const completed = completedSetupSteps(state);
  const steps: Array<{
    label: string;
    detail: string;
    complete: boolean;
    page: ConsolePage;
    action: string;
    icon: NovaIconName;
  }> = [
    { label: "连接画面", detail: "选择摄像头和实际支持的画面规格。", complete: state.captureReady, page: "capture", action: "设置画面", icon: "capture" },
    { label: "选择模型", detail: "检查模型资格并部署到当前设备。", complete: state.modelReady, page: "models", action: "选择模型", icon: "models" },
    { label: "确认参数", detail: "检查搜索范围、目标瞄点、触发延迟和跟随力度。", complete: state.configReady, page: "params", action: "检查参数", icon: "settings" },
    { label: "开始运行", detail: "回到首页核对状态，再由你明确启动。", complete: state.runtimeReady, page: "overview", action: "前往首页", icon: "start" },
  ];
  const next = steps.find((step) => !step.complete) ?? steps[SETUP_STEP_COUNT - 1]!;

  return (
    <section className="product-journey onboarding-view" aria-labelledby="onboarding-title">
      <header className="onboarding-hero">
        <div>
          <span>新手引导</span>
          <h2 id="onboarding-title">{statusKnown ? "完成首次设置" : "正在核对当前设备"}</h2>
          <p>{statusKnown ? "每一步都读取当前服务状态；你可以随时离开，已经完成的设置不会丢失。" : "服务状态尚不可用，暂时不能判断哪些步骤已经完成。请先检查连接。"}</p>
        </div>
        <div className="onboarding-progress" role="status" aria-label={statusKnown ? `已完成 ${completed} 步，共 ${SETUP_STEP_COUNT} 步` : "设置进度待核实"}>
          <strong>{statusKnown ? completed : "—"}<small>/{SETUP_STEP_COUNT}</small></strong>
          <span>{statusKnown ? "已完成" : "待核实"}</span>
        </div>
      </header>

      <ol className="onboarding-steps">
        {steps.map((step, index) => (
          <li className={!statusKnown ? "pending" : step.complete ? "complete" : step === next ? "current" : "pending"} key={step.label}>
            <span className="onboarding-step-index">{statusKnown && step.complete ? <NovaIcon name="check-circle" size={16} /> : index + 1}</span>
            <span className="onboarding-step-icon" aria-hidden="true"><NovaIcon name={step.icon} size={20} /></span>
            <div>
              <small>{!statusKnown ? "待核实" : step.complete ? "已完成" : step === next ? "下一步" : "稍后"}</small>
              <h3>{step.label}</h3>
              <p>{step.detail}</p>
            </div>
            <button className="console-button" onClick={() => onNavigate(step.page)} type="button">{step.action}</button>
          </li>
        ))}
      </ol>

      <footer className="onboarding-actions">
        {statusKnown ? <button className="console-button" onClick={() => onNavigate("overview")} type="button">跳过并返回首页</button> : null}
        <button className="console-button primary" onClick={() => onNavigate(statusKnown ? next.page : "activity")} type="button">{statusKnown ? next.action : "查看连接问题"}</button>
      </footer>
    </section>
  );
}

function runtimePhaseLabel(runtime: RuntimeState | null, verified: boolean): string {
  if (!runtime) return "等待状态";
  if (!verified) return "状态已过期";
  if (runtime.running) return "正在运行";
  if (runtime.semantic.phase === "faulted") return "运行故障";
  if (runtime.semantic.phase === "starting") return "正在启动";
  if (runtime.semantic.phase === "stopping") return "正在停止";
  return "已停止";
}

function metric(value: number | null | undefined, unit = ""): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(value >= 100 ? 0 : 1)}${unit}` : "等待样本";
}

export function DeviceStatusView({
  runtime,
  projection,
  captureProfile,
  activeModelName,
  modelLoaded,
  lastUpdated,
  desiredRevision,
  effectiveRevision,
  errorCount,
  onNavigate,
  onOpenErrors,
}: {
  runtime: RuntimeState | null;
  projection: RuntimeProjection | null;
  captureProfile: string;
  activeModelName: string;
  modelLoaded: boolean;
  lastUpdated: Date | null;
  desiredRevision: number;
  effectiveRevision: number;
  errorCount: number;
  onNavigate: Navigate;
  onOpenErrors: () => void;
}) {
  const kmnet = runtime?.executor.executors.kmnet;
  const verified = runtime !== null && projection?.transport === "current";
  const metricLabel = (value: number | null | undefined, unit: string) => verified && runtime?.statistics.metrics_available === true
    ? metric(value, unit)
    : verified ? "等待样本" : runtime ? "状态已过期" : "等待样本";
  const updated = lastUpdated?.toLocaleTimeString("zh-CN", { hour12: false }) ?? "尚未取得";
  return (
    <section className="product-journey device-status-view" aria-labelledby="device-truth-title">
      <header className="device-truth-header">
        <div>
          <span className={verified ? "truth-badge verified" : "truth-badge"}>
            <NovaIcon name={verified ? "shield-check" : "clock"} size={15} />
            {verified ? "当前状态已核实" : "等待当前状态"}
          </span>
          <h2 id="device-truth-title">{runtime?.capture.device || "当前设备"}</h2>
          <p>{verified ? `来自运行服务的最新完整状态，更新于 ${updated}。` : "收到完整运行状态前，不推断设备、识别或输出是否正常。"}</p>
        </div>
        <button className="console-button primary" onClick={() => onNavigate("capture")} type="button">查看实时画面</button>
      </header>

      <dl className="device-live-metrics" aria-label="设备实时数据">
        <div><dt>现在</dt><dd>{runtimePhaseLabel(runtime, verified)}</dd></div>
        <div><dt>推理输入</dt><dd>{metricLabel(runtime?.statistics.nvinfer_input_fps, " FPS")}</dd></div>
        <div><dt>推理耗时</dt><dd>{metricLabel(runtime?.statistics.inference_latency_ms, " ms")}</dd></div>
        <div><dt>结果新鲜度</dt><dd>{metricLabel(runtime?.statistics.detection_data_age_ms, " ms")}</dd></div>
      </dl>

      <section className="device-state-list" aria-label="设备各部分状态">
        <article>
          <span aria-hidden="true"><NovaIcon name="capture" size={19} /></span>
          <div><h3>画面</h3><p>{captureProfile || "尚未保存画面规格"}</p></div>
          <strong>{!verified ? "等待核实" : runtime?.capture.running ? "正在接收" : runtime?.capture.available ? "待运行" : "未确认"}</strong>
          <button onClick={() => onNavigate("capture")} type="button">设置</button>
        </article>
        <article>
          <span aria-hidden="true"><NovaIcon name="models" size={19} /></span>
          <div><h3>模型</h3><p>{activeModelName}</p></div>
          <strong>{!verified ? "等待核实" : modelLoaded ? "已装载" : runtime?.inference.configured ? "待装载" : "未配置"}</strong>
          <button onClick={() => onNavigate("models")} type="button">管理</button>
        </article>
        <article>
          <span aria-hidden="true"><NovaIcon name="device-send" size={19} /></span>
          <div><h3>设备输出</h3><p>{!verified ? "kmNet 连接状态待核实" : kmnet?.runtime_connected ? "kmNet 实时会话已连接" : "kmNet 未连接或等待主链"}</p></div>
          <strong>{projection?.output.label ?? "等待核实"}</strong>
          <button onClick={() => onNavigate("params")} type="button">查看</button>
        </article>
        <article>
          <span aria-hidden="true"><NovaIcon name="settings" size={19} /></span>
          <div><h3>配置</h3><p>{verified ? `保存版本 ${desiredRevision} · 生效版本 ${effectiveRevision}` : "等待运行态上报生效版本"}</p></div>
          <strong>{!verified ? "等待核实" : desiredRevision === effectiveRevision ? "版本一致" : "等待生效"}</strong>
          <button onClick={() => onNavigate("params")} type="button">检查</button>
        </article>
      </section>

      {errorCount > 0 ? (
        <button className="device-issue-link" onClick={onOpenErrors} type="button">
          <NovaIcon name="triangle-alert" size={17} />
          有 {errorCount} 条故障需要处理
        </button>
      ) : null}
    </section>
  );
}

export function ManagementView({
  license,
  deviceLabel,
  runtimeAvailable,
  projectCount,
  errorCount,
  setupState,
  onNavigate,
}: {
  license: LicenseStatus | null;
  deviceLabel: string;
  runtimeAvailable: boolean;
  projectCount: number;
  errorCount: number;
  setupState: SetupState;
  onNavigate: Navigate;
}) {
  const setupReady = setupState.captureReady && setupState.modelReady && setupState.configReady;
  const resources: Array<{ label: string; value: string; detail: string; page: ConsolePage; icon: NovaIconName }> = [
    { label: "设备", value: deviceLabel || (runtimeAvailable ? "尚未配置" : "待核实"), detail: runtimeAvailable ? "当前状态可读取" : "等待服务状态", page: "device", icon: "devices" },
    { label: "模型", value: projectCount > 0 || runtimeAvailable ? `${projectCount} 个项目` : "项目待核实", detail: !runtimeAvailable ? "等待服务状态" : setupState.modelReady ? "已有部署模型" : "尚未部署模型", page: "models", icon: "models" },
    { label: "活动", value: errorCount > 0 ? `${errorCount} 条需要注意` : "暂无故障记录", detail: "操作结果与故障历史", page: "activity", icon: "notification" },
    { label: "授权", value: license?.valid ? license.tier || "已授权" : license ? "需要授权" : "授权待核实", detail: license?.valid ? "当前权限有效" : license ? "当前权限不可用" : "等待授权信息", page: "license", icon: "shield-check" },
  ];
  return (
    <section className="product-journey management-view" aria-labelledby="management-title">
      <header className="management-hero">
        <div>
          <span>NovaSight 工作空间</span>
          <h2 id="management-title">设备与资源</h2>
          <p>查看设备连接，整理模型，或检查最近的操作记录。</p>
        </div>
        <button className="console-button primary" onClick={() => onNavigate(!runtimeAvailable || setupReady ? "device" : "onboarding")} type="button">
          {!runtimeAvailable || setupReady ? "查看设备状态" : "继续新手引导"}
        </button>
      </header>

      <div className="management-resources">
        {resources.map((resource) => (
          <button key={resource.label} onClick={() => onNavigate(resource.page)} type="button">
            <span aria-hidden="true"><NovaIcon name={resource.icon} size={20} /></span>
            <div><small>{resource.label}</small><strong>{resource.value}</strong><p>{resource.detail}</p></div>
            <NovaIcon name="forward" size={17} />
          </button>
        ))}
      </div>

      <section className="management-boundary" aria-labelledby="management-boundary-title">
        <div>
          <h3 id="management-boundary-title">当前由 Studio 管理</h3>
          <p>当前连接设备的模型、配置与授权相互独立。切换模型、恢复配置前会显示需要确认的改动。</p>
        </div>
        <button className="console-button" onClick={() => onNavigate("license")} type="button">查看授权范围</button>
      </section>
    </section>
  );
}

export function AboutView({
  license,
  serviceConnected,
  layoutRevision,
  modules,
  onNavigate,
}: {
  license: LicenseStatus | null;
  serviceConnected: boolean;
  layoutRevision: string;
  modules: readonly string[];
  onNavigate: Navigate;
}) {
  const moduleViews: Record<string, ReactNode> = {
    identity: (
      <header className="about-identity">
        <div className="about-mark" aria-hidden="true"><NovaIcon name="prediction-line" size={36} /></div>
        <div>
          <span>应用信息</span>
          <h2 id="about-product-title">NovaSight Studio</h2>
          <p>把采集、推理、目标选择与设备输出组织成一条可核实的实时链路。</p>
        </div>
        <span className={serviceConnected ? "truth-badge verified" : "truth-badge"}>
          <NovaIcon name={serviceConnected ? "connected" : "disconnected"} size={15} />
          {serviceConnected ? "服务已连接" : "服务未连接"}
        </span>
      </header>
    ),
    version: (
      <dl className="about-facts" aria-label="版本与配置信息">
        <div><dt>Studio 版本</dt><dd>{studioVersion}</dd></div>
        <div><dt>页面配置</dt><dd>{layoutRevision}</dd></div>
        <div><dt>授权</dt><dd>{license?.valid ? license.tier || "已授权" : license ? "未激活" : "待核实"}</dd></div>
      </dl>
    ),
    capabilities: (
      <section className="about-capabilities" aria-labelledby="about-capabilities-title">
        <div>
          <span>功能范围</span>
          <h3 id="about-capabilities-title">当前包含的功能</h3>
          <p>日常页面保持简单；需要排查时，可以继续查看 CUDA、TensorRT、NVMM 与原始运行证据。</p>
        </div>
        <ul>
          <li><NovaIcon name="capture" size={17} /><span>采集与 ROI</span></li>
          <li><NovaIcon name="tensorrt" size={17} /><span>模型与推理</span></li>
          <li><NovaIcon name="tracking" size={17} /><span>目标与跟踪</span></li>
          <li><NovaIcon name="device-send" size={17} /><span>控制与输出</span></li>
        </ul>
      </section>
    ),
    support: (
      <footer className="about-actions">
        <button className="console-button" onClick={() => onNavigate("license")} type="button">查看授权</button>
        <button className="console-button primary" onClick={() => onNavigate("activity")} type="button">打开实时日志</button>
        <a className="console-button" href="/third-party-ui.txt" target="_blank" rel="noreferrer">界面组件许可</a>
      </footer>
    ),
  };
  return (
    <section className="product-journey about-view" aria-labelledby="about-product-title">
      {modules.map((moduleId) => moduleViews[moduleId] ? <div data-module={moduleId} key={moduleId}>{moduleViews[moduleId]}</div> : null)}
    </section>
  );
}
