import { useState } from "react";
import type { RuntimeControlPipelineState } from "../../contracts/runtime";
import daylightArt from "../../assets/themes/nova-daylight.webp";
import "./prediction-insight.css";

type Sample = Partial<RuntimeControlPipelineState>;
const finite = (value: number | null | undefined) => typeof value === "number" && Number.isFinite(value) ? value : null;
const display = (value: number | null, unit: string) => value === null ? "—" : `${value.toFixed(1)} ${unit}`;

// Formula illustration only: no image geometry, closed-loop plant, or device output.
export function responseIllustration(kp: number, rampMs: number, referenceHz: number, elapsedMs: number, fps: number, errorCounts: number) {
  const q = rampMs === 0 ? 1 : Math.min(1, Math.max(0, elapsedMs / rampMs));
  const entry = q * q * (3 - 2 * q);
  const timeScale = referenceHz === 0 ? 1 : Math.min(50, 1000 / fps, Math.max(0, elapsedMs)) * referenceHz / 1000;
  return { entry, timeScale, demand: entry * timeScale * kp * 256 * Math.atan(errorCounts / 256) };
}

export function ResponseExperiment({ kp, rampMs, referenceHz }: { kp: number; rampMs: number; referenceHz: number }) {
  const [elapsed, setElapsed] = useState(100);
  const [fps, setFps] = useState(60);
  const [errorCounts, setErrorCounts] = useState(100);
  const endMs = Math.max(500, rampMs);
  const elapsedMs = Math.min(elapsed, endMs);
  const result = responseIllustration(kp, rampMs, referenceHz, elapsedMs, fps, errorCounts);
  const rampAt = (t: number) => responseIllustration(1, rampMs, 0, t, fps, 0).entry;
  const curve = Array.from({ length: 101 }, (_, i) => `${20 + i * 2.6},${120 - 100 * rampAt(endMs * i / 100)}`).join(" ");
  return <section className="prediction-insight" aria-label="响应试算" data-tone="active">
    <div className="prediction-measurement">
      <header><h3>响应试算</h3><span>编辑值 · 不控制设备</span></header>
      <p className="prediction-explanation">使用当前草稿中的跟随力度、入场时长和设备基准频率。下方模拟条件仅用于试算，不写入配置。</p>
      <div className="parameter-inline-grid">
        <label>入场经过 {elapsedMs} ms<input aria-label="试算入场时间" type="range" min="0" max={endMs} step="1" value={elapsedMs} onChange={(event) => setElapsed(Number(event.target.value))} /></label>
        <label>模拟控制频率<select aria-label="模拟控制频率" value={fps} onChange={(event) => setFps(Number(event.target.value))}>
          {[30, 60, 120, 240].map((value) => <option key={value} value={value}>{value} Hz</option>)}
        </select></label>
        <label>投影后误差 {errorCounts} counts<input aria-label="试算投影误差" type="range" min="-512" max="512" step="1" value={errorCounts} onChange={(event) => setErrorCounts(Number(event.target.value))} /></label>
      </div>
    </div>
    <div className="prediction-measurement">
      <div className="prediction-vector-row">
        <figure>
          <svg viewBox="0 0 300 150" role="img" aria-label={`入场曲线，当前达到设定 Kp 的 ${(result.entry * 100).toFixed(0)}%`}>
            <path className="prediction-grid" d="M20 20H280M20 70H280M20 120H280M20 20V120M150 20V120M280 20V120" />
            <polyline points={curve} fill="none" stroke="var(--signal-info)" strokeWidth="3" />
            <circle cx={20 + 260 * elapsedMs / endMs} cy={120 - result.entry * 100} r="5" fill="var(--ns-red-500)" />
          </svg>
          <figcaption>0 → {endMs} ms · 纵轴为 Kp 使用比例，不是速度曲线</figcaption>
        </figure>
        <dl><div><dt>当前 Kp 使用比例</dt><dd>{(result.entry * 100).toFixed(0)}%</dd></div>
          <div><dt>时间换算</dt><dd>×{result.timeScale.toFixed(2)}</dd></div>
          <div><dt>单次浮点需求</dt><dd><output aria-label="试算输出">{result.demand.toFixed(2)}</output> counts</dd></div></dl>
      </div>
      <p className="prediction-explanation">u = a × Kp × 256 × atan(c / 256){referenceHz > 0 ? " × min(Δt, 50 ms) × 基准 Hz / 1000" : ""}。仅示意单次需求，未计预测、整数化与发送限幅，不代表收敛速度或实机效果。</p>
    </div>
  </section>;
}

export function predictionInsight(sample: Sample | null | undefined, live: boolean) {
  const data = live ? sample : null;
  const x = finite(data?.prediction_safe_offset_x);
  const y = finite(data?.prediction_safe_offset_y);
  const rawX = finite(data?.prediction_raw_offset_x);
  const rawY = finite(data?.prediction_raw_offset_y);
  const offset = x === null || y === null ? null : Math.hypot(x, y);
  const raw = rawX === null || rawY === null ? null : Math.hypot(rawX, rawY);
  const age = finite(data?.frame_age_ms);
  const delay = finite(data?.prediction_actuation_delay_ms);
  const lead = finite(data?.prediction_lead_ms);
  const horizon = finite(data?.prediction_horizon_ms);
  const requested = age === null || delay === null || lead === null ? null : age + delay + lead;
  const ready = data?.prediction_allowed === true && data?.prediction_allowed_y === true;
  const unstable = data?.motion_state === "unstable" || data?.motion_state_y === "unstable";
  const clipped = ready && raw !== null && offset !== null && raw > offset + 0.01;
  const shortened = ready && requested !== null && horizon !== null && horizon < requested - 0.05;
  const incomplete = ready && (offset === null || raw === null || horizon === null);
  const status = !data ? "等待实时观测" : incomplete ? "运行数据不完整" : unstable ? "方向变化 · 暂停外推"
    : shortened ? "运行时域被缩短" : clipped ? "已到位移上限"
    : !ready ? "预测尚未生效" : offset !== null && offset < 0.1 ? "预测偏移很小" : "预测正在参与";
  const detail = !data ? "运行并触发后显示真实数据；这里不会用演示轨迹冒充实测。"
    : incomplete ? "预测字段未完整回读，暂不能判断预测贡献。"
    : unstable ? "新方向与历史不一致，当前不叠加预测偏移。"
    : shortened ? "运行时域小于年龄＋延迟＋提前量，请核对后端版本与运行配置。"
    : clipped ? "增加提前量也无法越过位移上限；先核对速度与标定。"
    : !ready ? "检查预测开关、触发条件，以及是否已有四个有效瞄点。"
    : offset !== null && offset < 0.1 ? "相对运动很小，开启与关闭接近是正常现象。"
    : "显示预测额外贡献，不代表已发送位移，也不代表预测精度。";
  return { x, y, rawX, rawY, offset, raw, age, delay, lead, horizon, requested, status, detail,
    tone: incomplete || unstable || clipped || shortened ? "attention" : ready ? "active" : "idle" };
}

export function PredictionInsight({ sample, live, configuredEnabled }: {
  sample?: Sample | null;
  live: boolean;
  configuredEnabled: boolean;
}) {
  const view = predictionInsight(sample, live);
  const extent = Math.max(10, view.raw ?? 0, view.offset ?? 0);
  const scale = 64 / extent;
  const endX = 150 + (view.x ?? 0) * scale;
  const endY = 80 + (view.y ?? 0) * scale;
  const total = Math.max(1, view.requested ?? 0);
  return (
    <section className="prediction-insight" aria-label="预测效果观察" data-tone={view.tone}>
      <div className="prediction-story">
        <img src={daylightArt} alt="" aria-hidden="true" width="1536" height="1024" loading="lazy" />
        <div><span className="prediction-edit-state">编辑值 · 预测{configuredEnabled ? "开启" : "关闭"}</span>
          <h3>提前多少，看得见</h3><p>观测位置与预测位置之间，才是预测真正增加的量。</p></div>
      </div>
      <div className="prediction-measurement">
        <header><h3>{view.status}</h3><span>运行回读</span></header>
        <div className="prediction-vector-row">
          <figure>
            <svg viewBox="0 0 300 160" role="img" aria-label={view.offset === null ? "等待预测位移数据" : `预测位移 ${view.offset.toFixed(1)} 像素，X ${view.x?.toFixed(1)}，Y ${view.y?.toFixed(1)}`}>
              <path className="prediction-grid" d="M30 40H270M30 80H270M30 120H270M70 20V140M150 20V140M230 20V140" />
              <circle className="prediction-envelope" cx="150" cy="80" r="64" />
              {view.rawX !== null && view.rawY !== null && <line className="prediction-raw" x1="150" y1="80" x2={150 + view.rawX * scale} y2={80 + view.rawY * scale} />}
              {view.offset !== null && <><line className="prediction-applied" x1="150" y1="80" x2={endX} y2={endY} /><path className="prediction-destination" d={`M${endX} ${endY - 6}l6 6-6 6-6-6Z`} /></>}
              <circle className="prediction-origin" cx="150" cy="80" r="4" />
            </svg>
            <figcaption><span className="prediction-observed-key">● 观测点</span><span className="prediction-future-key">◆ 预测点</span><span className="prediction-raw-key">┄ 原始外推</span><span>±{extent.toFixed(0)} px · Y 向下</span></figcaption>
          </figure>
          <dl><div><dt>实际预测偏移</dt><dd>{display(view.offset, "px")}</dd></div><div><dt>原始外推位移</dt><dd>{display(view.raw, "px")}</dd></div><div><dt>实际预测时域</dt><dd>{display(view.horizon, "ms")}</dd></div></dl>
        </div>
        <div className="prediction-time" aria-label="预测时间构成">
          <div className="prediction-time-bar" aria-hidden="true">
            {[view.age, view.delay, view.lead].map((value, i) => <i key={i} data-part={i} style={{ flexBasis: `${100 * Math.max(0, value ?? 0) / total}%` }} />)}
          </div>
          <div className="prediction-time-labels"><span>画面年龄 <b>{display(view.age, "ms")}</b></span><span>执行延迟 <b>{display(view.delay, "ms")}</b></span><span>额外提前 <b>{display(view.lead, "ms")}</b></span></div>
        </div>
        <p className="prediction-explanation">{view.detail}</p>
      </div>
    </section>
  );
}
