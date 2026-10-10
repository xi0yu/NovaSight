import { useCallback, useEffect, useState } from "react";

import {
  activityWebSocketUrl,
  clearActivityHistory,
  getActivityHistory,
  type ServerActivityEvent,
} from "../../api";
import { decodeActivityEventFrame } from "../../contracts/activity";
import { useWorkspaceAuthorization } from "../auth/AuthGate";

const HISTORY_LIMIT = 200;
const RECONNECT_MAX_MS = 15_000;
export type ActivityConnectionState = "loading" | "connected" | "disconnected" | "error" | "paused";

function mergeActivityEvent(current: ServerActivityEvent[], event: ServerActivityEvent): ServerActivityEvent[] {
  const key = `${event.daemon_instance_id}:${event.id}`;
  const next = current.filter((item) => `${item.daemon_instance_id}:${item.id}` !== key);
  next.push(event);
  next.sort((left, right) => left.occurred_at_ms - right.occurred_at_ms);
  return next.slice(-HISTORY_LIMIT);
}

export function useActivityStream() {
  const authorized = useWorkspaceAuthorization();
  const [events, setEvents] = useState<ServerActivityEvent[]>([]);
  const [state, setState] = useState<ActivityConnectionState>("loading");
  const [error, setError] = useState("");
  const [retryIndex, setRetryIndex] = useState(0);

  useEffect(() => {
    if (!authorized) { setState("paused"); return; }
    let active = true;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let reconnectAttempt = 0;
    setState("loading");
    setError("");

    void getActivityHistory().then((history) => {
      if (!active) return;
      setEvents((current) => history.events.reduce(mergeActivityEvent, current));
    }).catch((error: unknown) => {
      if (!active) return;
      setState("error");
      setError(error instanceof Error ? error.message : "日志历史未读取成功。");
    });

    const connect = () => {
      if (!active || typeof WebSocket === "undefined") return;
      let nextSocket: WebSocket;
      try {
        nextSocket = new WebSocket(activityWebSocketUrl());
      } catch {
        setState("error");
        setError("无法连接日志通道，正在自动重试。已有记录会保留。");
        const delay = Math.min(1_000 * 2 ** reconnectAttempt, RECONNECT_MAX_MS);
        reconnectAttempt += 1;
        reconnectTimer = window.setTimeout(connect, delay);
        return;
      }
      socket = nextSocket;
      nextSocket.onopen = () => {
        reconnectAttempt = 0;
        if (active) setState((current) => current === "error" ? current : "connected");
      };
      nextSocket.onmessage = (message) => {
        if (!active || socket !== nextSocket || typeof message.data !== "string") return;
        try {
          const frame = decodeActivityEventFrame(JSON.parse(message.data));
          setEvents((current) => mergeActivityEvent(current, frame.event));
          setState("connected");
          setError("");
        } catch {
          nextSocket.close(4002, "invalid activity contract");
        }
      };
      nextSocket.onclose = () => {
        if (!active || socket !== nextSocket) return;
        socket = null;
        setState("disconnected");
        setError("日志通道已断开，正在自动重试。已有记录不代表实时状态。");
        const delay = Math.min(1_000 * 2 ** reconnectAttempt, RECONNECT_MAX_MS);
        reconnectAttempt += 1;
        reconnectTimer = window.setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      active = false;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      socket?.close(1000, "activity stream suspended");
    };
  }, [authorized, retryIndex]);

  const clear = useCallback(async () => {
    await clearActivityHistory();
    setEvents([]);
  }, []);

  const retry = useCallback(() => setRetryIndex((index) => index + 1), []);
  return { events, clear, state, error, retry };
}
