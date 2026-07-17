import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import {
  activateMotionProfile,
  addMotionSample,
  createMotionSession,
  disableMotionProfile,
  getMotionProfileRuntime,
  getMotionProfiles,
  trainMotionProfile,
  type MotionProfile,
  type MotionSampleResult
} from "../../api";
import { getErrorMessage } from "../shared/format";
import {
  HumanTrajectoryCapture,
  TrainingTargetPlanner,
  pointerMovements,
  type TrainingTarget
} from "./trajectoryCapture";
import "./motion-profile.css";

type CanvasColors = { background: string; grid: string; target: string; ring: string; trace: string; cursor: string };

const CANVAS_WIDTH = 840;
const CANVAS_HEIGHT = 520;
const MIN_PROFILE_SAMPLES = 8;
const DEFAULT_CANVAS_COLORS: CanvasColors = {
  background: "#ffe8f1",
  grid: "rgba(126,24,66,.12)",
  target: "#b6195d",
  ring: "#fff",
  trace: "#4f1733",
  cursor: "#111827"
};

export function MotionProfileStudio() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureRef = useRef(new HumanTrajectoryCapture(CANVAS_WIDTH, CANVAS_HEIGHT));
  const targetPlannerRef = useRef(new TrainingTargetPlanner());
  const targetRef = useRef<TrainingTarget>({
    x: 650,
    y: 260,
    radius: 24,
    spawnedUs: Math.round(performance.now() * 1000),
    plannedGroup: "mid",
    plannedDirection: "right"
  });
  const animationRef = useRef(0);
  const canvasColorsRef = useRef<CanvasColors>(DEFAULT_CANVAS_COLORS);
  const canvasScaleRef = useRef({ x: 1, y: 1 });
  const uploadQueueRef = useRef<Promise<void>>(Promise.resolve());
  const [session, setSession] = useState<{ session_id: string } | null>(null);
  const [training, setTraining] = useState(false);
  const [locked, setLocked] = useState(false);
  const [capturedSamples, setCapturedSamples] = useState(0);
  const [savedSamples, setSavedSamples] = useState(0);
  const [validSamples, setValidSamples] = useState(0);
  const [lowQualitySamples, setLowQualitySamples] = useState(0);
  const [pendingUploads, setPendingUploads] = useState(0);
  const [profiles, setProfiles] = useState<MotionProfile[]>([]);
  const [activeProfile, setActiveProfile] = useState("");
  const [targetLabel, setTargetLabel] = useState("中距离 · 右侧");
  const [message, setMessage] = useState("准备开始训练");

  useEffect(() => {
    void getMotionProfiles().then(setProfiles).catch(() => undefined);
    void getMotionProfileRuntime().then((status) => setActiveProfile(status.active_profile)).catch(() => undefined);
  }, []);

  const beginNextTarget = useCallback(() => {
    const target = targetPlannerRef.current.next(captureRef.current.cursor, CANVAS_WIDTH, CANVAS_HEIGHT);
    targetRef.current = target;
    captureRef.current.begin(target);
    setTargetLabel(`${distanceGroupLabel(target.plannedGroup)} · ${directionLabel(target.plannedDirection)}`);
  }, []);

  const restartCurrentTarget = useCallback(() => {
    const target = { ...targetRef.current, spawnedUs: Math.round(performance.now() * 1000) };
    targetRef.current = target;
    captureRef.current.begin(target);
  }, []);

  useEffect(() => {
    const onLockChange = () => {
      const isLocked = document.pointerLockElement === canvasRef.current;
      setLocked(isLocked);
      if (!training) return;
      if (isLocked) {
        restartCurrentTarget();
        setMessage("采样已恢复；当前目标从此刻重新计时");
      } else {
        captureRef.current.cancel();
        setMessage("采样已暂停；点击训练画布后从当前目标重新开始");
      }
    };
    document.addEventListener("pointerlockchange", onLockChange);
    return () => document.removeEventListener("pointerlockchange", onLockChange);
  }, [restartCurrentTarget, training]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const updateScale = () => {
      const bounds = canvas.getBoundingClientRect();
      canvasScaleRef.current = {
        x: bounds.width > 0 ? canvas.width / bounds.width : 1,
        y: bounds.height > 0 ? canvas.height / bounds.height : 1
      };
    };
    updateScale();
    const observer = new ResizeObserver(updateScale);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const updateCanvasColors = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const styles = window.getComputedStyle(canvas);
      const read = (name: string, fallback: string) => styles.getPropertyValue(name).trim() || fallback;
      canvasColorsRef.current = {
        background: read("--motion-canvas-bg", DEFAULT_CANVAS_COLORS.background),
        grid: read("--motion-canvas-grid", DEFAULT_CANVAS_COLORS.grid),
        target: read("--motion-target-fill", DEFAULT_CANVAS_COLORS.target),
        ring: read("--motion-target-ring", DEFAULT_CANVAS_COLORS.ring),
        trace: read("--motion-trace", DEFAULT_CANVAS_COLORS.trace),
        cursor: read("--motion-cursor", DEFAULT_CANVAS_COLORS.cursor)
      };
    };
    updateCanvasColors();
    const observer = new MutationObserver(updateCanvasColors);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const draw = () => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (canvas && ctx) {
        const target = targetRef.current;
        const colors = canvasColorsRef.current;
        ctx.fillStyle = colors.background;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = colors.grid;
        ctx.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke(); }
        for (let y = 0; y < canvas.height; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }
        ctx.fillStyle = colors.target;
        ctx.beginPath();
        ctx.arc(target.x, target.y, target.radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = colors.ring;
        ctx.lineWidth = 2;
        ctx.stroke();
        const points = captureRef.current.points;
        if (points.length > 1) {
          ctx.strokeStyle = colors.trace;
          ctx.lineWidth = 3;
          ctx.beginPath();
          points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
          ctx.stroke();
        }
        const cursor = captureRef.current.cursor;
        ctx.strokeStyle = colors.cursor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cursor.x - 7, cursor.y);
        ctx.lineTo(cursor.x + 7, cursor.y);
        ctx.moveTo(cursor.x, cursor.y - 7);
        ctx.lineTo(cursor.x, cursor.y + 7);
        ctx.stroke();
      }
      animationRef.current = requestAnimationFrame(draw);
    };
    animationRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animationRef.current);
  }, []);

  const enqueueSample = useCallback((sessionId: string, payload: unknown) => {
    setPendingUploads((value) => value + 1);
    const upload = uploadQueueRef.current.then(() => addMotionSample(sessionId, payload));
    uploadQueueRef.current = upload.then(
      (result: MotionSampleResult) => {
        setSavedSamples((value) => value + 1);
        if (result.quality === "valid") {
          setValidSamples((value) => value + 1);
        } else {
          setLowQualitySamples((value) => value + 1);
          setMessage(`一条轨迹被标为低质量：${qualityReasonLabel(result.quality_reasons[0])}`);
        }
      },
      (error) => {
        setMessage(`轨迹保存失败：${getErrorMessage(error)}`);
      }
    ).finally(() => {
      setPendingUploads((value) => Math.max(0, value - 1));
    });
  }, []);

  const start = async () => {
    canvasRef.current?.requestPointerLock();
    await uploadQueueRef.current;
    try {
      const created = await createMotionSession("真人轨迹训练");
      setSession(created);
      setCapturedSamples(0);
      setSavedSamples(0);
      setValidSamples(0);
      setLowQualitySamples(0);
      targetPlannerRef.current.reset();
      setTraining(true);
      beginNextTarget();
      setMessage("按真实习惯移动并点击；系统会自动覆盖距离与方向");
    } catch (error) {
      setMessage(`训练会话创建失败：${getErrorMessage(error)}`);
    }
  };

  const onMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!training || document.pointerLockElement !== event.currentTarget) return;
    const scale = canvasScaleRef.current;
    captureRef.current.record(pointerMovements(
      event.nativeEvent,
      scale.x,
      scale.y
    ));
  };

  const onClick = () => {
    if (!training || !session) return;
    if (document.pointerLockElement !== canvasRef.current) {
      canvasRef.current?.requestPointerLock();
      return;
    }
    const target = targetRef.current;
    const cursor = captureRef.current.cursor;
    if (Math.hypot(cursor.x - target.x, cursor.y - target.y) > target.radius) {
      captureRef.current.markMiss();
      setMessage("未命中已记录，继续自然修正，不需要重新开始");
      return;
    }
    const payload = captureRef.current.finish(Math.round(performance.now() * 1000));
    if (!payload) return;
    setCapturedSamples((value) => value + 1);
    beginNextTarget();
    enqueueSample(session.session_id, payload);
    setMessage(`轨迹已采集；下一目标无需等待后端写盘`);
  };

  const train = async () => {
    if (!session) return;
    setMessage("正在等待采样落盘并生成画像…");
    await uploadQueueRef.current;
    try {
      const profile = await trainMotionProfile(session.session_id, "我的真人画像");
      setProfiles((items) => [profile, ...items]);
      setMessage(`画像已生成：${profile.name}，已分离反应时间和真实移动时间`);
    } catch (error) {
      setMessage(`画像生成失败：${getErrorMessage(error)}`);
    }
  };

  const activate = async (profileId: string) => {
    await activateMotionProfile(profileId);
    setActiveProfile(profileId);
    setMessage("真人曲线已在运行内存启用");
  };

  const disable = async () => {
    await disableMotionProfile();
    setActiveProfile("");
    setMessage("真人曲线已关闭，恢复静态算法配置");
  };

  return (
    <main className="motion-studio">
      <header>
        <div>
          <span className="eyebrow">NOVASIGHT / MOTION PROFILE</span>
          <h1>真人轨迹训练</h1>
          <p>高频事件展开、真实时间戳、异步落盘；压枪保持独立。</p>
        </div>
        <div className="motion-actions">
          <button className="motion-secondary" onClick={() => void disable()}>关闭真人曲线</button>
          <button className="motion-toggle" onClick={() => void start()}>{training ? "新建训练" : "开始训练"}</button>
        </div>
      </header>
      <section className="motion-grid">
        <div className="motion-card arena-card">
          <div className="card-head">
            <span>训练画布 · {locked ? "鼠标已锁定" : training ? "已暂停，点击画布恢复" : "等待开始"}</span>
            <strong>{targetLabel}</strong>
          </div>
          <canvas
            ref={canvasRef}
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
            tabIndex={0}
            onPointerMove={onMove}
            onClick={onClick}
          />
          <div className="motion-capture-strip">
            <span><b>{capturedSamples}</b> 已采集</span>
            <span><b>{savedSamples}</b> 已落盘</span>
            <span className={pendingUploads ? "pending" : "good"}><b>{pendingUploads}</b> 待保存</span>
            <span><b>{validSamples}</b> 有效</span>
            <span className={lowQualitySamples ? "warn" : ""}><b>{lowQualitySamples}</b> 低质量</span>
          </div>
          <div className="motion-status">{message}</div>
        </div>
        <aside className="motion-card profile-card">
          <div className="card-head"><span>画像工作区</span><span className="status-dot">● {training ? locked ? "采集中" : "暂停" : "待机"}</span></div>
          <div className="metric"><small>当前会话</small><b>{session?.session_id ?? "未开始"}</b></div>
          <div className="metric"><small>运行中画像</small><b>{activeProfile || "静态算法配置"}</b></div>
          <div className="metric"><small>采样质量</small><b>{validSamples} 有效 / {lowQualitySamples} 低质量</b></div>
          <div className="metric"><small>采样策略</small><b>4 距离 × 8 方向 × 3 目标尺寸</b></div>
          <button
            className="motion-secondary"
            disabled={!session || validSamples < MIN_PROFILE_SAMPLES || pendingUploads > 0}
            onClick={() => void train()}
          >
            {pendingUploads > 0 ? `等待 ${pendingUploads} 条落盘` : validSamples < MIN_PROFILE_SAMPLES ? `还需 ${MIN_PROFILE_SAMPLES - validSamples} 条有效轨迹` : "生成真人画像"}
          </button>
          <div className="profile-list">
            {profiles.map((profile) => (
              <div className="profile-row" key={profile.profile_id}>
                <span>{profile.name}</span>
                <em>{profile.sample_count} samples</em>
                <button disabled={activeProfile === profile.profile_id} onClick={() => void activate(profile.profile_id)}>
                  {activeProfile === profile.profile_id ? "使用中" : "启用"}
                </button>
              </div>
            ))}
          </div>
        </aside>
      </section>
    </main>
  );
}

function distanceGroupLabel(group: TrainingTarget["plannedGroup"]): string {
  return { micro: "微距离", near: "近距离", mid: "中距离", far: "远距离" }[group];
}

function directionLabel(direction: string): string {
  return {
    right: "右侧", left: "左侧", up: "上方", down: "下方",
    up_right: "右上", up_left: "左上", down_right: "右下", down_left: "左下"
  }[direction] ?? direction;
}

function qualityReasonLabel(reason: string | undefined): string {
  return {
    no_motion: "没有检测到有效移动",
    too_few_points: "有效轨迹点太少",
    dispatch_delay: "采样期间页面发生卡顿",
    boundary_hits: "虚拟准星多次碰到画布边缘",
    inefficient_path: "轨迹绕行过多",
    movement_too_short: "移动时间过短",
    movement_too_long: "移动时间过长",
    unusable_trajectory: "轨迹无法形成稳定参数"
  }[reason ?? ""] ?? reason ?? "采样不完整";
}
