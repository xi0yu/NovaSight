import { act, renderHook, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { useActivityStream } from "./useActivityStream";
const api = vi.hoisted(() => ({ getActivityHistory: vi.fn(), activityWebSocketUrl: vi.fn(() => "ws://localhost/ws/activity"), clearActivityHistory: vi.fn() }));
vi.mock("../../api", async (original) => ({ ...await original<typeof import("../../api")>(), ...api }));

it.each(["success", "failure"])("presents initial history and buffered stream events together on history %s", async (outcome) => {
  let finish!: (value: unknown) => void;
  let fail!: (error: Error) => void;
  api.getActivityHistory.mockImplementationOnce(() => new Promise((resolve, reject) => { finish = resolve; fail = reject; }));
  let socket!: FakeSocket;
  class FakeSocket {
    onopen: (() => void) | null = null;
    onclose: (() => void) | null = null;
    onmessage: ((message: { data: string }) => void) | null = null;
    constructor() { socket = this; }
    close() {}
  }
  vi.stubGlobal("WebSocket", FakeSocket);
  const event = { daemon_instance_id: "test", id: 1, occurred_at_ms: 1, level: "info", tag: "参数", title: "配置已保存", message: "saved", technical_detail: "", request_id: null, count: 1 };
  const { result, unmount } = renderHook(useActivityStream);
  act(() => {
    socket.onopen?.();
    for (const count of [1, 2]) socket.onmessage?.({ data: JSON.stringify({ kind: "activity_event", event: { ...event, count } }) });
  });
  expect(result.current.historyLoading).toBe(true);
  expect(result.current.events).toEqual([]);
  await act(async () => { if (outcome === "success") finish({ events: [event] }); else fail(new Error("history unavailable")); });
  expect(result.current.historyLoading).toBe(false);
  expect(result.current.events).toEqual([{ ...event, count: 2 }]);
  expect(result.current.state).toBe(outcome === "success" ? "connected" : "error");
  unmount();
});

it("exposes history failure, retries, and keeps records visible when the socket closes", async () => {
  const sockets: FakeSocket[] = [];
  class FakeSocket {
    onopen: (() => void) | null = null;
    onclose: (() => void) | null = null;
    onmessage: ((message: { data: string }) => void) | null = null;
    constructor() { sockets.push(this); }
    close() { this.onclose?.(); }
  }
  vi.stubGlobal("WebSocket", FakeSocket);
  api.getActivityHistory.mockRejectedValueOnce(new Error("history unavailable"));
  const event = { daemon_instance_id: "test", id: 1, occurred_at_ms: 1, level: "error", tag: "参数", title: "写入失败", message: "retry", technical_detail: "disk", request_id: "request-1", count: 1 };
  api.getActivityHistory.mockResolvedValueOnce({ events: [event] });
  const { result, unmount } = renderHook(useActivityStream);
  await waitFor(() => expect(result.current.state).toBe("error"));
  expect(result.current.error).toBe("history unavailable");
  act(() => result.current.retry());
  await waitFor(() => expect(result.current.events).toHaveLength(1));
  act(() => sockets[1].onopen?.());
  expect(result.current.state).toBe("connected");
  act(() => sockets[1].close());
  expect(result.current.state).toBe("disconnected");
  expect(result.current.events[0].technical_detail).toBe("disk");
  expect(result.current.error).toContain("已有记录不代表实时状态");
  unmount();
});

it("retries a failed constructor and a stalled connection, then recognizes recovery without new events", async () => {
  vi.useFakeTimers();
  let attempts = 0;
  const sockets: FakeSocket[] = [];
  class FakeSocket {
    onopen: (() => void) | null = null;
    onclose: (() => void) | null = null;
    onmessage: ((message: { data: string }) => void) | null = null;
    constructor() { if (attempts++ === 0) throw new Error("offline"); sockets.push(this); }
    close() { this.onclose?.(); }
  }
  vi.stubGlobal("WebSocket", FakeSocket);
  api.getActivityHistory.mockResolvedValue({ events: [] });
  const { result, unmount } = renderHook(useActivityStream);
  try {
    await act(async () => { await Promise.resolve(); });
    expect(result.current.state).toBe("error");
    act(() => vi.advanceTimersByTime(1000));
    act(() => vi.advanceTimersByTime(8000));
    expect(result.current.state).toBe("disconnected");
    act(() => vi.advanceTimersByTime(2000));
    act(() => sockets[1].onopen?.());
    expect(result.current.state).toBe("connected");
    expect(result.current.error).toBe("");
    expect(result.current.events).toEqual([]);
  } finally {
    unmount();
    vi.useRealTimers();
  }
});
