import { act, renderHook, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { useActivityStream } from "./useActivityStream";
const api = vi.hoisted(() => ({ getActivityHistory: vi.fn(), activityWebSocketUrl: vi.fn(() => "ws://localhost/ws/activity"), clearActivityHistory: vi.fn() }));
vi.mock("../../api", async (original) => ({ ...await original<typeof import("../../api")>(), ...api }));

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
