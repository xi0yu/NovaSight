import { Fragment, type ReactNode } from "react";

import { NovaIcon, type NovaIconName } from "../../components/visual";

function metricIconForTitle(title: string): NovaIconName {
  if (title.includes("FPS")) return "fps";
  if (title.includes("延迟") || title.includes("新鲜度") || title.includes("等待") || title.includes("帧间隔") || title.includes("端到端") || title.includes("队列")) return "latency";
  if (title.includes("分辨率")) return "resolution";
  if (title.includes("像素") || title.includes("格式")) return "frame";
  if (title.includes("丢帧") || title.includes("跳过")) return "signal-lost";
  if (title.includes("目标")) return "target";
  if (title.includes("引擎")) return "engine";
  if (title.includes("算法")) return "pid";
  if (title.includes("触发")) return "activity-pulse";
  if (title.includes("Kp")) return "gain";
  if (title.includes("角度") || title.includes("FOV")) return "fov";
  if (title.includes("限幅")) return "max-step";
  if (title.includes("运行时长")) return "clock";
  return "performance";
}

function metricToneForTitle(title: string): "target" | "compute" | "info" | "attention" | "neutral" {
  if (title.includes("目标") || title.includes("触发") || title.includes("控制量")) return "target";
  if (title.includes("FPS") || title.includes("GPU") || title.includes("推理")) return "compute";
  if (title.includes("延迟") || title.includes("新鲜度") || title.includes("帧间隔") || title.includes("端到端")) return "info";
  if (title.includes("丢帧") || title.includes("等待") || title.includes("队列") || title.includes("跳过")) return "attention";
  return "neutral";
}

function cardIconForTitle(title: string): NovaIconName {
  if (title.includes("采集")) return "capture";
  if (title.includes("推理")) return "inference";
  if (title.includes("系统")) return "system";
  if (title.includes("诊断")) return "triangle-alert";
  return "dashboard";
}

function sectionIconForTitle(title: string): NovaIconName {
  if (title.includes("采集设备")) return "capture-card";
  if (title.includes("ROI")) return "roi";
  if (title.includes("模型")) return "models";
  if (title.includes("推理输出")) return "output-tensor";
  if (title.includes("算法")) return "pid";
  if (title.includes("控制量")) return "output";
  if (title.includes("kmNet")) return "hid";
  if (title.includes("性能")) return "performance";
  if (title.includes("延迟")) return "latency";
  return "dashboard";
}

function sectionDescriptionForTitle(title: string): string {
  if (title.includes("采集设备")) return "视频源、后端通路与 latest-frame 策略";
  if (title.includes("ROI")) return "裁剪区域决定推理、预览和控制坐标基准";
  if (title.includes("模型设置")) return "绑定当前运行模型和 TensorRT 引擎";
  if (title.includes("推理输出")) return "查看 DetectionBatch、目标框和新鲜度";
  if (title.includes("鼠标移动算法")) return "PD/PID、滤波、预测和输出限幅";
  if (title.includes("控制量反馈")) return "控制决策、调度器和执行器状态";
  if (title.includes("kmNet")) return "硬件连接、触发键和移动诊断";
  if (title.includes("性能占比")) return "采集、预处理、推理和后处理耗时";
  if (title.includes("延迟链路")) return "按阶段定位实时链路瓶颈";
  if (title.includes("采集统计")) return "最新帧采集吞吐与丢弃情况";
  if (title.includes("推理统计")) return "批次消费、新鲜度和阶段耗时";
  if (title.includes("系统状态")) return "硬件资源和服务状态摘要";
  if (title.includes("采集诊断")) return "队列积压、帧间隔和采集建议";
  return "";
}

export function SectionTitle({ title, icon }: { title: string; icon?: NovaIconName }) {
  const description = sectionDescriptionForTitle(title);
  return (
    <h2 className="console-title">
      <span className="console-title-icon">
        <NovaIcon name={icon ?? sectionIconForTitle(title)} size={18} />
      </span>
      <span className="console-title-copy">
        <span>{title}</span>
        {description ? <small>{description}</small> : null}
      </span>
    </h2>
  );
}

export function Metric({
  title,
  value,
  small,
  icon
}: {
  title: string;
  value: string;
  small: string;
  icon?: NovaIconName;
}) {
  return (
    <div className={`console-metric tone-${metricToneForTitle(title)}`} aria-label={`${title}: ${value} ${small}`}>
      <span className="console-metric-head">
        <span className="console-metric-icon">
          <NovaIcon name={icon ?? metricIconForTitle(title)} size={18} />
        </span>
        <span>{title}</span>
      </span>
      <strong>{value}</strong>
      <small>{small}</small>
    </div>
  );
}

export function KvCard({ title, rows, notice }: { title: string; rows: [string, string][]; notice?: ReactNode }) {
  return (
    <div className="console-card">
      <SectionTitle icon={cardIconForTitle(title)} title={title} />
      {notice}
      <div className="console-kv">
        {rows.map(([key, value]) => (
          <Fragment key={key}>
            <span>{key}</span>
            <b>{value}</b>
          </Fragment>
        ))}
      </div>
    </div>
  );
}

export function Event({
  label,
  value,
  width
}: {
  label: string;
  value: string;
  width: number | null;
}) {
  const hasSample = width !== null && value !== "—";
  return (
    <div className={hasSample ? "console-event" : "console-event unavailable"}>
      <span>{label}</span>
      <div className="console-bar" aria-hidden="true">
        {hasSample ? <i style={{ width: `${Math.max(0, Math.min(100, width))}%` }} /> : null}
      </div>
      <b>{hasSample ? `${value} ms` : "暂无样本"}</b>
    </div>
  );
}
