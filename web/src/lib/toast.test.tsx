import { act, renderHook } from "@testing-library/react";
import { expect, it } from "vitest";
import { pushToastRaw, useClearErrorNotices, useErrorNotices, useToasts } from "./toast";

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
