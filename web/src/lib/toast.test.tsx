import { act, render, renderHook } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { ApiError, getRuntimeConfig } from "../api";
import { ToastHost } from "../components/ToastHost";
import { pushToastRaw, reportError, useClearErrorNotices, useDismissToast, useErrorNotices, useToasts } from "./toast";

it("keeps diagnostic records even when popup notifications are quiet", () => {
  const { result } = renderHook(() => ({ notices: useErrorNotices(), toasts: useToasts(), clear: useClearErrorNotices() }));
  act(() => result.current.clear());
  localStorage.setItem("ns-quiet-errors", "1");
  try {
    act(() => pushToastRaw({ tone: "error", title: "运行故障", detail: "GPU result batch", source: "runtime", status: null }));
    expect(result.current.notices).toHaveLength(1);
    expect(result.current.notices[0].detail).toBe("GPU result batch");
    expect(result.current.toasts).toHaveLength(0);
  } finally {
    localStorage.removeItem("ns-quiet-errors");
    act(() => result.current.clear());
  }
});

it("shows non-quiet failures immediately and retains the backend code for diagnostics", () => {
  const { result } = renderHook(() => ({ notices: useErrorNotices(), toasts: useToasts(), clear: useClearErrorNotices(), dismiss: useDismissToast() }));
  act(() => {
    result.current.clear();
    for (const toast of result.current.toasts) result.current.dismiss(toast.id);
  });
  const originalConsoleError = console.error;
  console.error = () => undefined;
  try {
    act(() => reportError(new ApiError("output rejected", 409, { code: "TEST_OUTPUT_REJECTED", message: "output rejected", license_key: "private" }), {
      source: "output-gate", title: "无法开启物理输出"
    }));
    expect(result.current.toasts[result.current.toasts.length - 1]?.detail).toBe("output rejected");
    expect(result.current.notices[result.current.notices.length - 1]?.technicalDetail).toContain("TEST_OUTPUT_REJECTED");
    expect(result.current.notices[result.current.notices.length - 1]?.technicalDetail).toContain('"http_status": 409');
    expect(result.current.notices[result.current.notices.length - 1]?.technicalDetail).not.toContain("private");
  } finally {
    console.error = originalConsoleError;
    act(() => {
      for (const toast of result.current.toasts) result.current.dismiss(toast.id);
      result.current.clear();
    });
  }
});

it("carries one safe request ID from API call into the retained error record", async () => {
  let sentId = "";
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
    sentId = new Headers(init?.headers).get("x-request-id") ?? "";
    return new Response(JSON.stringify({ code: "CONFIG_UNAVAILABLE", message: "configuration unavailable", license_key: "private" }), {
      status: 503,
      headers: { "content-type": "application/json", "x-request-id": sentId }
    });
  });
  const consoleMock = vi.spyOn(console, "error").mockImplementation(() => undefined);
  const { result } = renderHook(() => ({ notices: useErrorNotices(), clear: useClearErrorNotices() }));
  act(() => result.current.clear());
  try {
    expect.assertions(9);
    await getRuntimeConfig().catch((error: unknown) => {
      expect(sentId).toMatch(/^[0-9a-f]{32}$/);
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).requestId).toBe(sentId);
      act(() => reportError(error, { source: "config", popup: false }));
      expect(result.current.notices[0].requestId).toBe(sentId);
      expect(result.current.notices[0].technicalDetail).not.toContain("private");
      expect(consoleMock.mock.calls[0][0]).toMatchObject({ request_id: sentId, http_status: 503 });
      expect(JSON.stringify(consoleMock.mock.calls)).not.toContain("private");
      act(() => reportError(new ApiError((error as ApiError).message, 503, (error as ApiError).detail, "ffffffffffffffffffffffffffffffff"), {
        source: "config", popup: false
      }));
      expect(result.current.notices).toHaveLength(1);
      expect(result.current.notices[0]).toMatchObject({ count: 2, requestId: "ffffffffffffffffffffffffffffffff" });
    });
  } finally {
    fetchMock.mockRestore();
    consoleMock.mockRestore();
    act(() => result.current.clear());
  }
});

it("makes the request ID visible in a transient failure notice", () => {
  const { result } = renderHook(() => ({ toasts: useToasts(), dismiss: useDismissToast(), clear: useClearErrorNotices() }));
  const view = render(<ToastHost />);
  try {
    act(() => pushToastRaw({
      tone: "error", title: "操作失败", source: "request-id-test", status: 409,
      requestId: "0123456789abcdef0123456789abcdef"
    }));
    expect(view.getByText("排查编号 0123456789abcdef0123456789abcdef")).toBeVisible();
  } finally {
    act(() => {
      for (const toast of result.current.toasts) result.current.dismiss(toast.id);
      result.current.clear();
    });
    view.unmount();
  }
});

it("keeps background connection failures in diagnostics without covering controls", () => {
  const { result } = renderHook(() => ({ notices: useErrorNotices(), toasts: useToasts(), clear: useClearErrorNotices(), dismiss: useDismissToast() }));
  act(() => {
    for (const toast of result.current.toasts) result.current.dismiss(toast.id);
    result.current.clear();
    pushToastRaw({ tone: "warn", title: "实时通道异常", detail: "连接失败", source: "websocket:status", status: null }, false);
  });
  expect(result.current.notices).toHaveLength(1);
  expect(result.current.toasts).toHaveLength(0);
  act(() => result.current.clear());
});

it("clears only the error records the user agreed to remove", () => {
  const { result } = renderHook(() => ({ notices: useErrorNotices(), clear: useClearErrorNotices() }));
  act(() => {
    result.current.clear();
    pushToastRaw({ tone: "error", title: "旧故障", source: "old", status: null }, false);
  });
  const visibleIds = result.current.notices.map((notice) => notice.id);
  act(() => pushToastRaw({ tone: "error", title: "新故障", source: "new", status: null }, false));
  act(() => result.current.clear(visibleIds));
  expect(result.current.notices.map((notice) => notice.title)).toEqual(["新故障"]);
  act(() => result.current.clear());
});
