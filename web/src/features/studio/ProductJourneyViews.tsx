import type { LicenseStatus, RuntimeState } from "../../api";
import { NovaIcon, type NovaIconName } from "../../components/visual";
import type { RuntimeProjection } from "../runtime/runtimeProjection";
import type { ConsolePage } from "./StudioNavigation";

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

export function HomeSetupPrompt({ state, onNavigate }: { state: SetupState; onNavigate: Navigate }) {
  const completed = completedSetupSteps(state);
  if (completed === SETUP_STEP_COUNT) return null;
  return (
    <section className="home-setup-prompt" aria-labelledby="home-setup-title">
      <span className="home-setup-icon" aria-hidden="true"><NovaIcon name="play-circle" size={20} /></span>
      <div>
        <small>首次设置 · {completed}/{SETUP_STEP_COUNT}</small>
        <h2 id="home-setup-title">继续完成 NovaSight 设置</h2>
        <p>按顺序连接画面、选择模型、确认参数，再安全开始运行。</p>
      </div>
      <button className="console-button primary" onClick={() => onNavigate("onboarding")} type="button">
        继续设置
      </button>
    </section>
  );
}

export function OnboardingView({ state, onNavigate }: { state: SetupState; onNavigate: Navigate }) {
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
    { label: "确认参数", detail: "确认响应方式、移动方式和输出保持暂停。", complete: state.configReady, page: "params", action: "检查参数", icon: "settings" },
    { label: "开始运行", detail: "回到首页核对状态，再由你明确启动。", complete: state.runtimeReady, page: "overview", action: "前往首页", icon: "start" },
  ];
  const next = steps.find((step) => !step.complete) ?? steps[SETUP_STEP_COUNT - 1]!;

  return (
    <section className="product-journey onboarding-view" aria-labelledby="onboarding-title">
      <header className="onboarding-hero">
        <div>
          <span>新手引导</span>
          <h2 id="onboarding-title">四步准备好第一条视觉链路</h2>
          <p>每一步都读取当前服务状态；你可以随时离开，已经完成的设置不会丢失。</p>
        </div>
        <div className="onboarding-progress" role="status" aria-label={`已完成 ${completed} 步，共 ${SETUP_STEP_COUNT} 步`}>
          <strong>{completed}<small>/{SETUP_STEP_COUNT}</small></strong>
          <span>已完成</span>
        </div>
      </header>

      <ol className="onboarding-steps">
        {steps.map((step, index) => (
          <li className={step.complete ? "complete" : step === next ? "current" : "pending"} key={step.label}>
            <span className="onboarding-step-index">{step.complete ? <NovaIcon name="check-circle" size={16} /> : index + 1}</span>
            <span className="onboarding-step-icon" aria-hidden="true"><NovaIcon name={step.icon} size={20} /></span>
            <div>
              <small>{step.complete ? "已完成" : step === next ? "下一步" : "稍后"}</small>
              <h3>{step.label}</h3>
              <p>{step.detail}</p>
            </div>
            <button className="console-button" onClick={() => onNavigate(step.page)} type="button">{step.action}</button>
          </li>
        ))}
      </ol>

      <footer className="onboarding-actions">
        <button className="console-button" onClick={() => onNavigate("overview")} type="button">跳过并返回首页</button>
        <button className="console-button primary" onClick={() => onNavigate(next.page)} type="button">{next.action}</button>
      </footer>
    </section>
  );
}

function runtimePhaseLabel(runtime: RuntimeState | null): string {
  if (!runtime) return "等待状态";
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
        <div><dt>现在</dt><dd>{runtimePhaseLabel(runtime)}</dd></div>
        <div><dt>推理输入</dt><dd>{metric(runtime?.statistics.nvinfer_input_fps, " FPS")}</dd></div>
        <div><dt>推理耗时</dt><dd>{metric(runtime?.statistics.inference_latency_ms, " ms")}</dd></div>
        <div><dt>结果新鲜度</dt><dd>{metric(runtime?.statistics.detection_data_age_ms, " ms")}</dd></div>
      </dl>

      <section className="device-state-list" aria-label="设备各部分状态">
        <article>
          <span aria-hidden="true"><NovaIcon name="capture" size={19} /></span>
          <div><h3>画面</h3><p>{captureProfile || "尚未保存画面规格"}</p></div>
          <strong>{runtime?.capture.running ? "正在接收" : runtime?.capture.available ? "待运行" : "未确认"}</strong>
          <button onClick={() => onNavigate("capture")} type="button">设置</button>
        </article>
        <article>
          <span aria-hidden="true"><NovaIcon name="models" size={19} /></span>
          <div><h3>模型</h3><p>{activeModelName}</p></div>
          <strong>{!runtime ? "等待核实" : modelLoaded ? "已装载" : runtime.inference.configured ? "待装载" : "未配置"}</strong>
          <button onClick={() => onNavigate("models")} type="button">管理</button>
        </article>
        <article>
          <span aria-hidden="true"><NovaIcon name="device-send" size={19} /></span>
          <div><h3>设备输出</h3><p>{kmnet?.runtime_connected ? "kmNet 实时会话已连接" : "kmNet 未连接或等待主链"}</p></div>
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
  const setupComplete = completedSetupSteps(setupState) === SETUP_STEP_COUNT;
  const resources: Array<{ label: string; value: string; detail: string; page: ConsolePage; icon: NovaIconName }> = [
    { label: "设备", value: deviceLabel || "尚未配置", detail: runtimeAvailable ? "当前状态可读取" : "等待服务状态", page: "device", icon: "devices" },
    { label: "模型", value: `${projectCount} 个项目`, detail: setupState.modelReady ? "已有部署模型" : "尚未部署模型", page: "models", icon: "models" },
    { label: "活动", value: errorCount > 0 ? `${errorCount} 条需要注意` : "没有待处理故障", detail: "操作结果与故障历史", page: "activity", icon: "notification" },
    { label: "授权", value: license?.valid ? license.tier || "已授权" : "需要授权", detail: license?.valid ? "当前权限有效" : "当前权限不可用", page: "license", icon: "shield-check" },
  ];
  return (
    <section className="product-journey management-view" aria-labelledby="management-title">
      <header className="management-hero">
        <div>
          <span>NovaSight 工作空间</span>
          <h2 id="management-title">管理真实对象，不管理后台术语</h2>
          <p>设备、模型、活动和授权集中在这里。每项数据都来自当前服务，不用虚构的 KPI 填满页面。</p>
        </div>
        <button className="console-button primary" onClick={() => onNavigate(setupComplete ? "device" : "onboarding")} type="button">
          {setupComplete ? "查看设备状态" : "继续新手引导"}
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
          <p>这里管理本机设备、模型、配置与授权。团队、账单和多设备云端编排没有后端数据时不会伪装成可用功能。</p>
        </div>
        <button className="console-button" onClick={() => onNavigate("license")} type="button">查看授权范围</button>
      </section>
    </section>
  );
}
