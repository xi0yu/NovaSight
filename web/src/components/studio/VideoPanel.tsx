import { useEffect, useMemo, useState, type ReactNode } from "react";

import { StatusIndicator } from "../ui";

type VideoPanelProps = {
  src?: string;
  available: boolean;
  running?: boolean;
  profile?: ReactNode;
  caption?: ReactNode;
  className?: string;
  children?: ReactNode;
};

type VideoState = {
  tone: "good" | "warn" | "bad" | "idle";
  title: string;
  detail: string;
};

export function VideoPanel({
  src,
  available,
  running = false,
  profile,
  caption,
  className,
  children
}: VideoPanelProps) {
  const [loaded, setLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    setLoaded(false);
    setHasError(false);
  }, [src]);

  const videoState = useMemo<VideoState | null>(() => {
    if (!available) {
      return {
        tone: "idle",
        title: "采集未打开",
        detail: "应用采集配置后，预览会在同一个采集会话上恢复。"
      };
    }
    if (hasError) {
      return {
        tone: "bad",
        title: "预览流加载失败",
        detail: "浏览器没有收到最新 MJPEG 帧，请检查采集后端或刷新能力。"
      };
    }
    if (!loaded) {
      return {
        tone: "warn",
        title: "正在连接预览流",
        detail: running ? "等待浏览器收到最新帧。" : "等待后端开始输出预览帧。"
      };
    }
    return null;
  }, [available, hasError, loaded, running]);

  const classes = className ? `video-shell ${className}` : "video-shell";

  return (
    <div className={classes}>
      {src ? (
        <img
          alt="实时采集画面"
          src={src}
          onError={() => {
            setHasError(true);
            setLoaded(false);
          }}
          onLoad={() => {
            setLoaded(true);
            setHasError(false);
          }}
        />
      ) : (
        <div className="video-panel-placeholder" aria-hidden="true" />
      )}
      <div className="scan-lines" />
      <div className="reticle" />
      <div className="corner-frame corner-frame-tl" />
      <div className="corner-frame corner-frame-tr" />
      <div className="corner-frame corner-frame-bl" />
      <div className="corner-frame corner-frame-br" />
      {videoState ? (
        <div className="video-overlay">
          <div className={`video-panel-state tone-${videoState.tone}`}>
            <StatusIndicator tone={videoState.tone}>{videoState.title}</StatusIndicator>
            <span>{videoState.detail}</span>
          </div>
        </div>
      ) : null}
      {profile ? (
        <div className="video-status">
          <StatusIndicator tone={available ? "good" : "idle"}>
            {available ? "采集中" : "未打开"}
          </StatusIndicator>
          <span className="mono">{profile}</span>
        </div>
      ) : null}
      {caption ? <div className="preview-caption">{caption}</div> : null}
      {children}
    </div>
  );
}
