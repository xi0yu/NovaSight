import { useCallback, useEffect, useState } from "react";

import {
  activityWebSocketUrl,
  clearActivityHistory,
  getActivityHistory,
  type ServerActivityEvent,
} from "../../api";
import { decodeActivityEventFrame } from "../../contracts/activity";

const HISTORY_LIMIT = 200;
const RECONNECT_MAX_MS = 15_000;

function mergeActivityEvent(current: ServerActivityEvent[], event: ServerActivityEvent): ServerActivityEvent[] {
  const key = `${event.daemon_instance_id}:${event.id}`;
  const next = current.filter((item) => `${item.daemon_instance_id}:${item.id}` !== key);
  next.push(event);
  next.sort((left, right) => left.occurred_at_ms - right.occurred_at_ms);
  return next.slice(-HISTORY_LIMIT);
}

export function useActivityStream(): { events: ServerActivityEvent[]; clear: () => Promise<void> } {
  const [events, setEvents] = useState<ServerActivityEvent[]>([]);

  useEffect(() => {
    let active = true;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let reconnectAttempt = 0;

    void getActivityHistory().then((history) => {
      if (!active) return;
      setEvents((current) => history.events.reduce(mergeActivityEvent, current));
    }).catch(() => undefined);

    const connect = () => {
      if (!active || typeof WebSocket === "undefined") return;
      let nextSocket: WebSocket;
      try {
        nextSocket = new WebSocket(activityWebSocketUrl());
      } catch {
        const delay = Math.min(1_000 * 2 ** reconnectAttempt, RECONNECT_MAX_MS);
        reconnectAttempt += 1;
        reconnectTimer = window.setTimeout(connect, delay);
        return;
      }
      socket = nextSocket;
      nextSocket.onopen = () => {
        reconnectAttempt = 0;
      };
      nextSocket.onmessage = (message) => {
        if (!active || socket !== nextSocket || typeof message.data !== "string") return;
        try {
          const frame = decodeActivityEventFrame(JSON.parse(message.data));
          setEvents((current) => mergeActivityEvent(current, frame.event));
        } catch {
          nextSocket.close(4002, "invalid activity contract");
        }
      };
      nextSocket.onclose = () => {
        if (!active || socket !== nextSocket) return;
        socket = null;
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
  }, []);

  const clear = useCallback(async () => {
    await clearActivityHistory();
    setEvents([]);
  }, []);

  return { events, clear };
}
