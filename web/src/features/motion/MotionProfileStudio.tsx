import { useEffect, useRef, useState } from "react";
import {
  activateMotionProfile,
  addMotionSample,
  createMotionSession,
  disableMotionProfile,
  getMotionProfileRuntime,
  getMotionProfiles,
  trainMotionProfile,
  type MotionProfile
} from "../../api";
import "./motion-profile.css";

type Point = { t_us: number; x: number; y: number; dx: number; dy: number };
type Target = { x: number; y: number; radius: number; spawnedUs: number };
type CanvasColors = { background: string; grid: string; target: string; ring: string; trace: string; cursor: string };

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
  const pointsRef = useRef<Point[]>([]);
  const cursorRef = useRef({ x: 420, y: 260 });
  const targetRef = useRef<Target>({ x: 650, y: 260, radius: 28, spawnedUs: performance.now() * 1000 });
  const animationRef = useRef(0);
  const canvasColorsRef = useRef<CanvasColors>(DEFAULT_CANVAS_COLORS);
  const [session, setSession] = useState<{ session_id: string } | null>(null);
  const [training, setTraining] = useState(false);
  const [locked, setLocked] = useState(false);
  const [samples, setSamples] = useState(0);
  const [profiles, setProfiles] = useState<MotionProfile[]>([]);
  const [activeProfile, setActiveProfile] = useState("");
  const [message, setMessage] = useState("准备开始训练");

  useEffect(() => {
    void getMotionProfiles().then(setProfiles).catch(() => undefined);
    void getMotionProfileRuntime().then((status) => setActiveProfile(status.active_profile)).catch(() => undefined);
  }, []);

  const spawnTarget = () => {
    const spawnedUs = Math.round(performance.now() * 1000);
    targetRef.current = {
      x: 50 + Math.random() * 740,
      y: 50 + Math.random() * 420,
      radius: 20 + Math.random() * 18,
      spawnedUs
    };
    const cursor = cursorRef.current;
    pointsRef.current = [{ t_us: spawnedUs, x: cursor.x, y: cursor.y, dx: 0, dy: 0 }];
  };

  useEffect(() => {
    const onLockChange = () => setLocked(document.pointerLockElement === canvasRef.current);
    document.addEventListener("pointerlockchange", onLockChange);
    return () => document.removeEventListener("pointerlockchange", onLockChange);
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
        ctx.beginPath(); ctx.arc(target.x, target.y, target.radius, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = colors.ring; ctx.lineWidth = 2; ctx.stroke();
        const points = pointsRef.current;
        if (points.length > 1) {
          ctx.strokeStyle = colors.trace; ctx.lineWidth = 3; ctx.beginPath();
          points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
          ctx.stroke();
        }
        const cursor = cursorRef.current;
        ctx.strokeStyle = colors.cursor; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(cursor.x - 7, cursor.y); ctx.lineTo(cursor.x + 7, cursor.y); ctx.moveTo(cursor.x, cursor.y - 7); ctx.lineTo(cursor.x, cursor.y + 7); ctx.stroke();
      }
      animationRef.current = requestAnimationFrame(draw);
    };
    animationRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animationRef.current);
  }, []);

  const start = async () => {
    canvasRef.current?.requestPointerLock();
    const created = await createMotionSession("真人轨迹训练");
    setSession(created); setSamples(0); setTraining(true);
    spawnTarget(); setMessage("鼠标已锁定；移动虚拟准星并点击目标");
  };

  const onMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!training || document.pointerLockElement !== event.currentTarget) return;
    const cursor = cursorRef.current;
    cursor.x = Math.max(0, Math.min(event.currentTarget.width, cursor.x + event.movementX));
    cursor.y = Math.max(0, Math.min(event.currentTarget.height, cursor.y + event.movementY));
    pointsRef.current.push({ t_us: Math.round(performance.now() * 1000), x: cursor.x, y: cursor.y, dx: event.movementX, dy: event.movementY });
  };

  const onClick = async () => {
    if (!training || !session) return;
    if (!locked) { canvasRef.current?.requestPointerLock(); return; }
    const target = targetRef.current;
    const cursor = cursorRef.current;
    if (Math.hypot(cursor.x - target.x, cursor.y - target.y) > target.radius) {
      setMessage("未命中，继续修正"); return;
    }
    const clickUs = Math.round(performance.now() * 1000);
    const points = [...pointsRef.current, { t_us: clickUs, x: cursor.x, y: cursor.y, dx: 0, dy: 0 }];
    await addMotionSample(session.session_id, {
      spawn_x: points[0]?.x ?? cursor.x,
      spawn_y: points[0]?.y ?? cursor.y,
      target_x: target.x,
      target_y: target.y,
      radius_px: target.radius,
      target_spawn_us: target.spawnedUs,
      click_us: clickUs,
      points,
      quality: points.length >= 5 ? "valid" : "low_quality"
    });
    const count = samples + 1;
    setSamples(count); spawnTarget();
    setMessage(`第 ${count} 条轨迹已记录`);
  };

  const train = async () => {
    if (!session) return;
    const profile = await trainMotionProfile(session.session_id, "我的真人画像");
    setProfiles((items) => [profile, ...items]);
    setMessage(`画像已生成：${profile.name}，Fitts 与进度曲线已拟合`);
  };

  const activate = async (profileId: string) => {
    await activateMotionProfile(profileId); setActiveProfile(profileId); setMessage("真人曲线已在运行内存启用");
  };

  const disable = async () => {
    await disableMotionProfile(); setActiveProfile(""); setMessage("真人曲线已关闭，恢复静态算法配置");
  };

  return <main className="motion-studio">
    <header><div><span className="eyebrow">NOVASIGHT / MOTION PROFILE</span><h1>真人轨迹训练</h1><p>训练硬件触发后的目标移动节奏；压枪保持独立。</p></div><div className="motion-actions"><button className="motion-secondary" onClick={() => void disable()}>关闭真人曲线</button><button className="motion-toggle" onClick={() => void start()}>{training ? "重新训练" : "开始训练"}</button></div></header>
    <section className="motion-grid">
      <div className="motion-card arena-card"><div className="card-head"><span>训练画布 · {locked ? "鼠标已锁定" : "点击画布锁定鼠标"}</span><strong>{samples.toString().padStart(2, "0")} 条样本</strong></div><canvas ref={canvasRef} width={840} height={520} tabIndex={0} onPointerMove={onMove} onClick={() => void onClick()} /><div className="motion-status">{message}</div></div>
      <aside className="motion-card profile-card"><div className="card-head"><span>画像工作区</span><span className="status-dot">● {training ? "采集中" : "待机"}</span></div><div className="metric"><small>当前会话</small><b>{session?.session_id ?? "未开始"}</b></div><div className="metric"><small>运行中画像</small><b>{activeProfile || "静态算法配置"}</b></div><div className="metric"><small>训练建议</small><b>覆盖不同距离和方向，至少 30 条</b></div><button className="motion-secondary" disabled={!session || samples < 3} onClick={() => void train()}>生成真人画像</button><div className="profile-list">{profiles.map((profile) => <div className="profile-row" key={profile.profile_id}><span>{profile.name}</span><em>{profile.sample_count} samples</em><button disabled={activeProfile === profile.profile_id} onClick={() => void activate(profile.profile_id)}>{activeProfile === profile.profile_id ? "使用中" : "启用"}</button></div>)}</div></aside>
    </section>
  </main>;
}
