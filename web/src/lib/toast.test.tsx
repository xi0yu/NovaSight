import { act, renderHook } from "@testing-library/react";
import { expect, it } from "vitest";
import { ApiError } from "../api";
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
