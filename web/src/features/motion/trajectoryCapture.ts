export type TrajectoryPoint = { t_us: number; x: number; y: number; dx: number; dy: number };

export type PointerMovement = { time_ms: number; dx: number; dy: number; coalesced: boolean; dispatch_delay_ms: number };

export type TrainingTarget = {
  x: number;
  y: number;
  radius: number;
  spawnedUs: number;
  plannedGroup: "micro" | "near" | "mid" | "far";
  plannedDirection: string;
};

export type CapturedTrajectory = {
  spawn_x: number;
  spawn_y: number;
  target_x: number;
  target_y: number;
  radius_px: number;
  target_spawn_us: number;
  first_motion_us: number | null;
  click_us: number;
  points: TrajectoryPoint[];
  quality: "pending";
  capture: {
    event_count: number;
    coalesced_event_count: number;
    dropped_event_count: number;
    max_event_gap_ms: number;
    max_dispatch_delay_ms: number;
    boundary_hit_count: number;
    miss_click_count: number;
    canvas_width: number;
    canvas_height: number;
    device_pixel_ratio: number;
    planned_distance_group: string;
    planned_direction: string;
  };
};

const MAX_POINTS_PER_SAMPLE = 4096;
const MOTION_START_DISTANCE_PX = 1.5;
const DISTANCE_PLANS = [
  { group: "micro", min: 28, max: 55 },
  { group: "near", min: 75, max: 140 },
  { group: "mid", min: 175, max: 290 },
  { group: "far", min: 320, max: 430 }
] as const;
const DIRECTION_PLANS = [
  { name: "right", angle: 0 },
  { name: "left", angle: Math.PI },
  { name: "down", angle: Math.PI / 2 },
  { name: "up", angle: -Math.PI / 2 },
  { name: "down_right", angle: Math.PI / 4 },
  { name: "up_left", angle: -3 * Math.PI / 4 },
  { name: "down_left", angle: 3 * Math.PI / 4 },
  { name: "up_right", angle: -Math.PI / 4 }
] as const;

export class HumanTrajectoryCapture {
  private activeTarget: TrainingTarget | null = null;
  private startCursor = { x: 420, y: 260 };
  private currentCursor = { x: 420, y: 260 };
  private capturedPoints: TrajectoryPoint[] = [];
  private firstMotionUs: number | null = null;
  private lastEventUs = 0;
  private eventCount = 0;
  private coalescedEventCount = 0;
  private droppedEventCount = 0;
  private maxEventGapUs = 0;
  private maxDispatchDelayMs = 0;
  private boundaryHitCount = 0;
  private missClickCount = 0;

  constructor(private readonly width: number, private readonly height: number) {}

  get cursor(): Readonly<{ x: number; y: number }> {
    return this.currentCursor;
  }

  get points(): readonly TrajectoryPoint[] {
    return this.capturedPoints;
  }

  begin(target: TrainingTarget): void {
    this.activeTarget = target;
    this.startCursor = { ...this.currentCursor };
    this.capturedPoints = [{
      t_us: target.spawnedUs,
      x: this.currentCursor.x,
      y: this.currentCursor.y,
      dx: 0,
      dy: 0
    }];
    this.firstMotionUs = null;
    this.lastEventUs = target.spawnedUs;
    this.eventCount = 0;
    this.coalescedEventCount = 0;
    this.droppedEventCount = 0;
    this.maxEventGapUs = 0;
    this.maxDispatchDelayMs = 0;
    this.boundaryHitCount = 0;
    this.missClickCount = 0;
  }

  cancel(): void {
    this.activeTarget = null;
    this.capturedPoints = [];
    this.firstMotionUs = null;
  }

  record(movements: readonly PointerMovement[]): void {
    if (!this.activeTarget) return;
    for (const movement of movements) {
      if (!Number.isFinite(movement.dx) || !Number.isFinite(movement.dy)) continue;
      if (movement.dx === 0 && movement.dy === 0) continue;
      const requestedX = this.currentCursor.x + movement.dx;
      const requestedY = this.currentCursor.y + movement.dy;
      const nextX = Math.max(0, Math.min(this.width, requestedX));
      const nextY = Math.max(0, Math.min(this.height, requestedY));
      const actualDx = nextX - this.currentCursor.x;
      const actualDy = nextY - this.currentCursor.y;
      if (actualDx !== movement.dx || actualDy !== movement.dy) this.boundaryHitCount += 1;
      this.currentCursor = { x: nextX, y: nextY };
      this.eventCount += 1;
      if (movement.coalesced) this.coalescedEventCount += 1;
      this.maxDispatchDelayMs = Math.max(this.maxDispatchDelayMs, movement.dispatch_delay_ms);
      const eventUs = Math.max(this.lastEventUs + 1, Math.round(movement.time_ms * 1000));
      this.maxEventGapUs = Math.max(this.maxEventGapUs, eventUs - this.lastEventUs);
      this.lastEventUs = eventUs;
      if (this.firstMotionUs === null && Math.hypot(nextX - this.startCursor.x, nextY - this.startCursor.y) >= MOTION_START_DISTANCE_PX) {
        this.firstMotionUs = eventUs;
      }
      const point = { t_us: eventUs, x: nextX, y: nextY, dx: actualDx, dy: actualDy };
      if (this.capturedPoints.length < MAX_POINTS_PER_SAMPLE) {
        this.capturedPoints.push(point);
      } else {
        this.droppedEventCount += 1;
        const last = this.capturedPoints[this.capturedPoints.length - 1];
        this.capturedPoints[this.capturedPoints.length - 1] = {
          ...point,
          dx: last.dx + actualDx,
          dy: last.dy + actualDy
        };
      }
    }
  }

  markMiss(): void {
    this.missClickCount += 1;
  }

  finish(clickUs: number): CapturedTrajectory | null {
    const target = this.activeTarget;
    if (!target) return null;
    const safeClickUs = Math.max(this.lastEventUs + 1, Math.round(clickUs));
    const points = [...this.capturedPoints, {
      t_us: safeClickUs,
      x: this.currentCursor.x,
      y: this.currentCursor.y,
      dx: 0,
      dy: 0
    }];
    const result: CapturedTrajectory = {
      spawn_x: this.startCursor.x,
      spawn_y: this.startCursor.y,
      target_x: target.x,
      target_y: target.y,
      radius_px: target.radius,
      target_spawn_us: target.spawnedUs,
      first_motion_us: this.firstMotionUs,
      click_us: safeClickUs,
      points,
      quality: "pending",
      capture: {
        event_count: this.eventCount,
        coalesced_event_count: this.coalescedEventCount,
        dropped_event_count: this.droppedEventCount,
        max_event_gap_ms: this.maxEventGapUs / 1000,
        max_dispatch_delay_ms: this.maxDispatchDelayMs,
        boundary_hit_count: this.boundaryHitCount,
        miss_click_count: this.missClickCount,
        canvas_width: this.width,
        canvas_height: this.height,
        device_pixel_ratio: window.devicePixelRatio || 1,
        planned_distance_group: target.plannedGroup,
        planned_direction: target.plannedDirection
      }
    };
    this.activeTarget = null;
    return result;
  }
}

export class TrainingTargetPlanner {
  private sampleIndex = 0;
  private readonly bucketCounts = new Map<string, number>();
  private readonly groupCounts = new Map<string, number>();
  private readonly directionCounts = new Map<string, number>();

  constructor(private readonly random: () => number = Math.random) {}

  reset(): void {
    this.sampleIndex = 0;
    this.bucketCounts.clear();
    this.groupCounts.clear();
    this.directionCounts.clear();
  }

  next(cursor: Readonly<{ x: number; y: number }>, width: number, height: number): TrainingTarget {
    const radius = [18, 24, 30][Math.floor(this.sampleIndex / 8) % 3];
    const margin = radius + 8;
    const candidates: Array<{ x: number; y: number; group: TrainingTarget["plannedGroup"]; direction: string; score: number }> = [];
    const addCandidate = (rawX: number, rawY: number) => {
      const x = Math.max(margin, Math.min(width - margin, rawX));
      const y = Math.max(margin, Math.min(height - margin, rawY));
      const dx = x - cursor.x;
      const dy = y - cursor.y;
      const distance = Math.hypot(dx, dy);
      if (distance < 24) return;
      const group = actualDistanceGroup(distance);
      const direction = actualDirection(dx, dy);
      const bucket = `${group}:${direction}`;
      const score =
        (this.bucketCounts.get(bucket) ?? 0) * 10_000
        + (this.groupCounts.get(group) ?? 0) * 240
        + (this.directionCounts.get(direction) ?? 0) * 18
        + this.random();
      candidates.push({ x, y, group, direction, score });
    };
    for (const distance of DISTANCE_PLANS) {
      for (const direction of DIRECTION_PLANS) {
        const magnitude = distance.min + this.random() * (distance.max - distance.min);
        const angle = direction.angle + (this.random() - 0.5) * 0.16;
        addCandidate(cursor.x + Math.cos(angle) * magnitude, cursor.y + Math.sin(angle) * magnitude);
      }
    }
    for (let index = 0; index < 96; index += 1) {
      addCandidate(margin + this.random() * Math.max(1, width - margin * 2), margin + this.random() * Math.max(1, height - margin * 2));
    }
    const selected = candidates.reduce((best, item) => item.score < best.score ? item : best, candidates[0]);
    const bucket = `${selected.group}:${selected.direction}`;
    this.bucketCounts.set(bucket, (this.bucketCounts.get(bucket) ?? 0) + 1);
    this.groupCounts.set(selected.group, (this.groupCounts.get(selected.group) ?? 0) + 1);
    this.directionCounts.set(selected.direction, (this.directionCounts.get(selected.direction) ?? 0) + 1);
    this.sampleIndex += 1;
    return {
      x: selected.x,
      y: selected.y,
      radius,
      spawnedUs: Math.round(performance.now() * 1000),
      plannedGroup: selected.group,
      plannedDirection: selected.direction
    };
  }
}

export function pointerMovements(event: PointerEvent, scaleX = 1, scaleY = 1): PointerMovement[] {
  const coalesced = typeof event.getCoalescedEvents === "function" ? event.getCoalescedEvents() : [];
  const source = coalesced.length > 0 ? coalesced : [event];
  const dispatchTime = performance.now();
  return source.map((item) => {
    const timeMs = normalizeEventTime(item.timeStamp);
    return {
      time_ms: timeMs,
      dx: item.movementX * scaleX,
      dy: item.movementY * scaleY,
      coalesced: coalesced.length > 0,
      dispatch_delay_ms: Math.max(0, dispatchTime - timeMs)
    };
  });
}

function actualDistanceGroup(distance: number): TrainingTarget["plannedGroup"] {
  return distance < 50 ? "micro" : distance < 150 ? "near" : distance < 350 ? "mid" : "far";
}

function actualDirection(x: number, y: number): string {
  if (Math.abs(x) >= Math.abs(y) * 1.5) return x >= 0 ? "right" : "left";
  if (Math.abs(y) >= Math.abs(x) * 1.5) return y >= 0 ? "down" : "up";
  return `${y >= 0 ? "down" : "up"}_${x >= 0 ? "right" : "left"}`;
}

function normalizeEventTime(timeStamp: number): number {
  if (!Number.isFinite(timeStamp)) return performance.now();
  const relative = timeStamp > performance.timeOrigin ? timeStamp - performance.timeOrigin : timeStamp;
  const now = performance.now();
  return Math.max(0, Math.min(now + 5, relative));
}
