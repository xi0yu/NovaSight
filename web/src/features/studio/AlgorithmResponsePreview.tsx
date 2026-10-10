import { useId } from "react";
import { NovaIcon } from "../../components/visual";
import { responseIllustration } from "./PredictionInsight";

export function StartupResponsePreview({ delayMs, rampMs }: { delayMs: number; rampMs: number }) {
  const endMs = Math.max(1000, Math.ceil((delayMs + rampMs + 200) / 500) * 500);
  const x = (time: number) => 36 + 292 * time / endMs;
  const points = Array.from(new Set([0, delayMs, delayMs + rampMs, endMs,
    ...Array.from({ length: 81 }, (_, i) => delayMs + rampMs * i / 80)]))
    .sort((a, b) => a - b)
    .map((time) => `${x(time)},${time < delayMs ? 112 : 112 - 80 * responseIllustration(1, rampMs, 0, time - delayMs, 60, 0).entry}`)
    .join(" ");
  const curve = rampMs > 0 ? `M${points.replace(/ /g, " L")}` : `M36 112H${x(delayMs)}V32H328`;
  return <figure className="algorithm-response-preview startup-curve">
    <svg viewBox="0 0 356 156" role="img" aria-label={`启动响应曲线：等待 ${delayMs} ms，渐增 ${rampMs} ms，${delayMs + rampMs} ms 后达到完整力度`}>
      <path className="response-grid" d="M36 32H328M36 72H328M36 112H328M36 32V112" />
      <rect className="response-delay-band" x="36" y="32" width={x(delayMs) - 36} height="80" />
      <path className="response-boundary" d={`M${x(delayMs)} 24V120M${x(delayMs + rampMs)} 24V120`} />
      <path className="response-curve" d={curve} />
      <text x="28" y="36" textAnchor="end">100%</text><text x="28" y="116" textAnchor="end">0%</text>
      <text x="36" y="138">0</text><text x="328" y="138" textAnchor="end">{endMs} ms</text>
    </svg>
    <figcaption><span>触发等待 <b>{delayMs} ms</b></span><span>力度渐增 <b>{rampMs} ms</b></span></figcaption>
    <p>示意：等待结束时已有有效目标。</p>
  </figure>;
}

export function SingleOutputPreview({ kp, limitX, limitY }: { kp: number; limitX: number; limitY: number }) {
  const plotClipId = useId();
  const demand = (error: number) => responseIllustration(kp, 0, 0, 0, 60, error).demand;
  const clamp = (value: number, limit: number) => Math.max(-Math.floor(limit), Math.min(Math.floor(limit), value));
  const extent = Math.max(1, Math.ceil(Math.max(Math.floor(limitX), Math.floor(limitY)) * 1.1));
  const capBoundary = (limit: number) => Math.abs(demand(512)) <= Math.floor(limit)
    ? undefined : 256 * Math.tan(Math.floor(limit) / (kp * 256));
  const curve = (limit?: number) => {
    const boundary = limit === undefined ? undefined : capBoundary(limit);
    const errors = Array.from({ length: 101 }, (_, i) => -512 + 1024 * i / 100);
    if (boundary !== undefined) errors.push(-boundary, boundary);
    return errors.sort((a, b) => a - b).map((error) => {
      const raw = demand(error);
      return `${182 + 292 * error / 1024},${88 - 56 * (limit === undefined ? raw : clamp(raw, limit)) / extent}`;
    }).join(" ");
  };
  const limitedCurve = (limit: number) => {
    const cap = Math.floor(limit);
    // Invert Atan so the highlight starts exactly where demand reaches the cap.
    const boundary = capBoundary(limit);
    if (boundary === undefined) return undefined;
    return `M36 ${88 + 56 * cap / extent}H${182 - 292 * boundary / 1024}M${182 + 292 * boundary / 1024} ${88 - 56 * cap / extent}H328`;
  };
  const limitedX = limitedCurve(limitX);
  const limitedY = limitedCurve(limitY);
  const sample = demand(100);
  return <figure className="algorithm-response-preview single-output-preview" aria-label="单次 Kp 输出示意">
    <ol className="single-output-flow" aria-label="单次输出计算流程">
      <li><NovaIcon name="target" size={18} /><span>投影误差<b>100 counts</b></span></li>
      <li><NovaIcon name="prediction-line" size={18} /><span>Kp 响应<b>{sample.toFixed(2)} counts</b></span></li>
      <li><NovaIcon name="shield-check" size={18} /><span>X / Y 限幅<b>{clamp(sample, limitX).toFixed(2)} / {clamp(sample, limitY).toFixed(2)}</b></span></li>
    </ol>
    <svg viewBox="0 0 356 176" role="img" aria-label={`单次响应曲线：Kp ${kp}，X 上限 ${limitX}，Y 上限 ${limitY} counts，纵轴范围 ±${extent} counts`}>
      <defs><clipPath id={plotClipId}><rect x="36" y="32" width="292" height="112" /></clipPath></defs>
      <path className="response-grid" d="M36 32H328M36 88H328M36 144H328M36 32V144M182 32V144" />
      <g clipPath={`url(#${plotClipId})`}>
        <polyline className="response-raw" points={curve()} />
        <polyline className="response-x" points={curve(limitX)} />
        <polyline className="response-y" points={curve(limitY)} />
        {limitedX && <path className="response-limited" d={limitedX} aria-label="X 轴受上限限制"><title>X 轴已达 ±{Math.floor(limitX)} counts 上限</title></path>}
        {limitedY && <path className="response-limited response-limited-y" d={limitedY} aria-label="Y 轴受上限限制"><title>Y 轴已达 ±{Math.floor(limitY)} counts 上限</title></path>}
      </g>
      <text x="28" y="36" textAnchor="end">{extent}</text><text x="28" y="92" textAnchor="end">0</text><text x="28" y="148" textAnchor="end">−{extent}</text>
      <text x="36" y="164">−512</text><text x="182" y="164" textAnchor="middle">0</text><text x="328" y="164" textAnchor="end">512 counts</text>
    </svg>
    <figcaption><span className="raw-key">浮点需求</span><span className="x-key">X 轴输出（实线）</span><span className="y-key">Y 轴输出（虚线）</span><span className="limited-key">琥珀色段：受上限限制</span></figcaption>
    <p>纵轴由输出上限固定；灰色需求超出图表范围时隐藏。单次静态示意，不含预测与整数余量，不发送指令。</p>
  </figure>;
}
