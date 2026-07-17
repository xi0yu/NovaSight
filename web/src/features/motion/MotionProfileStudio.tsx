import { useEffect, useRef, useState } from "react";
import { createMotionSession, addMotionSample, trainMotionProfile, getMotionProfiles, type MotionProfile } from "../../api";
import "./motion-profile.css";

type Point = { t_us: number; x: number; y: number; dx: number; dy: number };

export function MotionProfileStudio() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [session, setSession] = useState<{ session_id: string; sample_count?: number } | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [target, setTarget] = useState({ x: 420, y: 260, radius: 28 });
  const [training, setTraining] = useState(false);
  const [samples, setSamples] = useState(0);
  const [profiles, setProfiles] = useState<MotionProfile[]>([]);
  const [message, setMessage] = useState("准备开始训练");

  useEffect(() => { void getMotionProfiles().then(setProfiles).catch(() => undefined); }, []);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ffe8f1"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "rgba(126,24,66,.12)"; ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke(); }
    for (let y = 0; y < canvas.height; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }
    ctx.fillStyle = "#b6195d"; ctx.beginPath(); ctx.arc(target.x, target.y, target.radius, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
    if (points.length > 1) { ctx.strokeStyle = "#4f1733"; ctx.lineWidth = 3; ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.stroke(); }
  }, [points, target]);

  const start = async () => {
    const created = await createMotionSession("真人轨迹训练");
    setSession(created); setSamples(0); setTraining(true); setPoints([]); setMessage("移动到目标并点击");
    const canvas = canvasRef.current; canvas?.focus();
  };
  const onMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!training) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left; const y = event.clientY - rect.top;
    setPoints((current) => [...current, { t_us: Math.round(performance.now() * 1000), x, y, dx: event.movementX, dy: event.movementY }].slice(-800));
  };
  const onClick = async (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!training || !session) return;
    const rect = event.currentTarget.getBoundingClientRect(); const x = event.clientX - rect.left; const y = event.clientY - rect.top;
    const hit = Math.hypot(x - target.x, y - target.y) <= target.radius;
    if (!hit) { setMessage("点击未命中，继续尝试"); return; }
    await addMotionSample(session.session_id, { spawn_x: target.x, spawn_y: target.y, target_x: target.x, target_y: target.y, radius_px: target.radius, points, quality: points.length > 2 ? "valid" : "low_quality" });
    const count = samples + 1; setSamples(count); setPoints([]); setMessage(`已记录第 ${count} 条轨迹`);
    setTarget({ x: 50 + Math.random() * 740, y: 50 + Math.random() * 420, radius: 20 + Math.random() * 18 });
  };
  const train = async () => { if (!session) return; const profile = await trainMotionProfile(session.session_id, "我的真人画像"); setProfiles((items) => [profile, ...items]); setMessage(`画像已生成：${profile.name}`); };
  return <main className="motion-studio">
    <header><div><span className="eyebrow">NOVASIGHT / MOTION PROFILE</span><h1>真人轨迹训练</h1><p>只记录硬件触发前后的目标移动风格；压枪仍由原控制链独立负责。</p></div><button className="motion-toggle" onClick={() => void start()}>{training ? "重新开始" : "开始训练"}</button></header>
    <section className="motion-grid"><div className="motion-card arena-card"><div className="card-head"><span>训练画布</span><strong>{samples.toString().padStart(2, "0")} 条样本</strong></div><canvas ref={canvasRef} width={840} height={520} tabIndex={0} onPointerMove={onMove} onClick={(event) => void onClick(event)} /><div className="motion-status">{message}</div></div>
      <aside className="motion-card profile-card"><div className="card-head"><span>画像工作区</span><span className="status-dot">● {training ? "采集中" : "待机"}</span></div><div className="metric"><small>当前会话</small><b>{session?.session_id ?? "未开始"}</b></div><div className="metric"><small>训练建议</small><b>至少 30 条有效轨迹</b></div><button className="motion-secondary" disabled={!session || samples < 3} onClick={() => void train()}>生成真人画像</button><div className="profile-list">{profiles.map((profile) => <div className="profile-row" key={profile.profile_id}><span>{profile.name}</span><em>{profile.sample_count} samples</em></div>)}</div></aside>
    </section>
  </main>;
}
