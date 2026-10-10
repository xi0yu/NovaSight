import { act, render, renderHook } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { ApiError, getRuntimeConfig } from "../api";
import { ToastHost } from "../components/ToastHost";
import { pushToastRaw, reportError, reportSuccess, useActivityNotices, useClearErrorNotices, useDismissToast, useErrorNotices, useToasts } from "./toast";

it("retains completed operations for the Activity page", () => {
  const { result } = renderHook(() => ({
    activity: useActivityNotices(),
    toasts: useToasts(),
    dismiss: useDismissToast(),
  }));
  act(() => {
    for (const toast of result.current.toasts) result.current.dismiss(toast.id);
  });
  act(() => reportSuccess("模型已更新", "新模型已经部署。", "model-switch"));
  expect(result.current.activity[result.current.activity.length - 1]).toMatchObject({ tone: "success", title: "模型已更新", detail: "新模型已经部署。" });
  act(() => {
    for (const toast of result.current.toasts) result.current.dismiss(toast.id);
  });
});

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

it("explains a browser connection failure while retaining its original detail", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new TypeError("Failed to fetch"));
  const consoleMock = vi.spyOn(console, "error").mockImplementation(() => undefined);
  const { result } = renderHook(() => ({ notices: useErrorNotices(), clear: useClearErrorNotices() }));
  act(() => result.current.clear());
  try {
    const error = await getRuntimeConfig().catch((failure: unknown) => failure);
    expect(error).toBeInstanceOf(TypeError);
    act(() => reportError(error, { source: "config", popup: false }));
    expect(result.current.notices[0].detail).toContain("无法连接 NovaSight 服务");
    expect(result.current.notices[0].technicalDetail).toContain("Failed to fetch");
    expect(result.current.notices[0].status).toBeNull();
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
    const details = view.getByText("排查信息").closest("details")!;
    expect(details).not.toHaveAttribute("open");
    expect(details).toHaveTextContent("0123456789abcdef0123456789abcdef");
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

it("shows red validation failure and green completion feedback even in quiet mode", () => {
  const { result } = renderHook(() => ({ toasts: useToasts(), dismiss: useDismissToast(), clear: useClearErrorNotices() }));
  localStorage.setItem("ns-quiet-errors", "1");
  const consoleMock = vi.spyOn(console, "error").mockImplementation(() => undefined);
  try {
    act(() => {
      for (const toast of result.current.toasts) result.current.dismiss(toast.id);
      reportError(new ApiError("invalid parameters", 422, { code: "CONFIG_REJECTED" }), { source: "parameter-page", title: "参数自动保存失败", tone: "error", popup: "feedback" });
      reportSuccess("参数已自动保存并应用", "saved", "parameter-page", "feedback");
    });
    expect(result.current.toasts.map((toast) => toast.tone)).toEqual(["error", "success"]);
  } finally {
    localStorage.removeItem("ns-quiet-errors");
    consoleMock.mockRestore();
    act(() => { for (const toast of result.current.toasts) result.current.dismiss(toast.id); result.current.clear(); });
  }
});

it("delivers every explicit action feedback while deduplicating passive notices", () => {
  const { result } = renderHook(() => ({ toasts: useToasts(), dismiss: useDismissToast() }));
  try {
    act(() => {
      for (const toast of result.current.toasts) result.current.dismiss(toast.id);
      reportSuccess("协议已启用", "same device", "protocol-feedback-test", "feedback");
      reportSuccess("协议已启用", "same device", "protocol-feedback-test", "feedback");
      reportSuccess("后台状态更新", "same status", "passive-feedback-test");
      reportSuccess("后台状态更新", "same status", "passive-feedback-test");
    });
    expect(result.current.toasts.map((toast) => toast.title)).toEqual(["协议已启用", "协议已启用", "后台状态更新"]);
  } finally {
    act(() => { for (const toast of result.current.toasts) result.current.dismiss(toast.id); });
  }
});

it("expires explicit failure feedback but retains its diagnostic record", () => {
  vi.useFakeTimers();
  const consoleMock = vi.spyOn(console, "error").mockImplementation(() => undefined);
  const { result } = renderHook(() => ({ toasts: useToasts(), notices: useErrorNotices(), dismiss: useDismissToast(), clear: useClearErrorNotices() }));
  try {
    act(() => {
      for (const toast of result.current.toasts) result.current.dismiss(toast.id);
      result.current.clear();
      reportError(new Error("IP invalid"), { source: "protocol-validation-expiry", title: "协议参数无效", popup: "feedback" });
    });
    expect(result.current.toasts).toHaveLength(1);
    act(() => vi.advanceTimersByTime(2999));
    expect(result.current.toasts).toHaveLength(1);
    act(() => vi.advanceTimersByTime(1));
    expect(result.current.toasts).toHaveLength(0);
    expect(result.current.notices).toHaveLength(1);
    expect(result.current.notices[0].detail).toBe("IP invalid");
  } finally {
    act(() => { for (const toast of result.current.toasts) result.current.dismiss(toast.id); result.current.clear(); });
    consoleMock.mockRestore();
    vi.useRealTimers();
  }
});

it.each(["info", "warn", "error", "success"] as const)("shows a three-second countdown then removes a %s toast", (tone) => {
  vi.useFakeTimers();
  const { result } = renderHook(() => ({ toasts: useToasts(), clear: useClearErrorNotices(), dismiss: useDismissToast() }));
  const view = render(<ToastHost />);
  try {
    act(() => {
      for (const toast of result.current.toasts) result.current.dismiss(toast.id);
      pushToastRaw({ tone, title: "倒计时测试", source: `countdown-${tone}`, status: null });
    });
    expect(view.getByText("3 秒后关闭")).toBeVisible();
    expect(view.container.querySelector(".toast-card-timer > span")).toHaveStyle({ animationDuration: "3000ms", animationDelay: "0ms" });
    act(() => vi.advanceTimersByTime(1500));
    expect(view.getByText("2 秒后关闭")).toBeVisible();
    act(() => vi.advanceTimersByTime(1499));
    expect(view.getByText("倒计时测试")).toBeVisible();
    act(() => vi.advanceTimersByTime(1));
    expect(view.queryByText("倒计时测试")).not.toBeInTheDocument();
  } finally {
    act(() => { for (const toast of result.current.toasts) result.current.dismiss(toast.id); result.current.clear(); });
    view.unmount();
    vi.useRealTimers();
  }
});
