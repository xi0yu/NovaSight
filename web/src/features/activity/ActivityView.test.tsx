import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";

import { ActivityView } from "./ActivityView";

it("separates completed operations from items that need attention", () => {
  const { rerender } = render(<ActivityView items={[{
    key: "success-1",
    title: "模型已更新",
    detail: "新模型已经部署。",
    tone: "success",
    time: Date.now(),
  }]} onClear={vi.fn()} onOpenDetails={vi.fn()} />);

  expect(screen.getByRole("heading", { name: "最近操作已完成" })).toBeVisible();
  expect(screen.queryByRole("button", { name: "查看完整详情" })).not.toBeInTheDocument();

  rerender(<ActivityView items={[{
    key: "error-1",
    title: "运行故障",
    detail: "推理链已停止。",
    technicalDetail: "backend detail",
    tone: "error",
  }]} onClear={vi.fn()} onOpenDetails={vi.fn()} />);

  expect(screen.getByRole("heading", { name: "1 条故障与警告记录" })).toBeVisible();
  expect(screen.getByRole("button", { name: "查看完整详情" })).toBeVisible();
  expect(screen.getByText("原始错误与开发者详情")).toBeVisible();
});

it("does not call an empty activity list healthy", () => {
  render(<ActivityView items={[]} onClear={vi.fn()} onOpenDetails={vi.fn()} />);
  expect(screen.getByRole("heading", { name: "暂无活动记录" })).toBeVisible();
  expect(screen.queryByText("一切正常")).not.toBeInTheDocument();
});

it("renders records as a list and exposes clearing", async () => {
  const onClear = vi.fn();
  render(<ActivityView items={[{
    key: "info-1",
    title: "配置已保存",
    detail: "后端已确认。",
    tone: "info",
  }]} onClear={onClear} onOpenDetails={vi.fn()} />);

  expect(screen.getByRole("list", { name: "最近活动" })).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "清理历史" }));
  expect(onClear).toHaveBeenCalledOnce();
});

it("searches diagnostic IDs and preserves records after a synchronous clear failure", async () => {
  render(<ActivityView items={[{ key: "a", title: "保存失败", detail: "待核对", requestId: "trace-001", tone: "error" }]}
    onClear={() => { throw new Error("无权清理"); }} onOpenDetails={vi.fn()} />);
  await userEvent.type(screen.getByRole("searchbox", { name: "搜索日志" }), "trace-001");
  expect(screen.getByText("保存失败")).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "清理历史" }));
  expect(screen.getByRole("alert")).toHaveTextContent("无权清理");
  expect(screen.getByRole("button", { name: "清理历史" })).toBeEnabled();
  expect(screen.getByText("时间未提供")).toBeVisible();
  await userEvent.type(screen.getByRole("searchbox"), "missing");
  await userEvent.click(screen.getByRole("button", { name: "重置筛选" }));
  expect(screen.getByText("保存失败")).toBeVisible();
});

it("pauses the viewed records while new events arrive and retains source filters on resume", async () => {
  const items = [{ key: "one", title: "参数已保存", detail: "已应用", tone: "success" as const, tag: "参数" }, { key: "two", title: "模型已部署", detail: "已确认", tone: "info" as const, tag: "模型" }];
  const props = { onClear: vi.fn(), onOpenDetails: vi.fn() };
  const view = render(<ActivityView {...props} items={items} />);
  await userEvent.selectOptions(screen.getByRole("combobox", { name: "日志来源" }), "参数");
  expect(screen.queryByText("模型已部署")).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "暂停查看" }));
  view.rerender(<ActivityView {...props} items={[{ key: "new", title: "参数写入失败", detail: "等待重试", tone: "error", tag: "参数" }, ...items]} />);
  expect(screen.queryByText("参数写入失败")).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "恢复实时 · 1 条新记录" }));
  expect(screen.getByText("参数写入失败")).toBeVisible();
  expect(screen.getByRole("combobox", { name: "日志来源" })).toHaveValue("参数");
  await userEvent.type(screen.getByRole("searchbox", { name: "搜索日志" }), "missing");
  await userEvent.click(screen.getByRole("button", { name: "重置筛选" }));
  expect(screen.getByRole("combobox", { name: "日志来源" })).toHaveValue("all");
  expect(screen.getByText("模型已部署")).toBeVisible();
});

it("freezes the viewed records when scrolling away and resumes at the latest records", async () => {
  const item = { key: "one", title: "原有记录", detail: "待检查", tone: "info" as const };
  const props = { onClear: vi.fn(), onOpenDetails: vi.fn() };
  const view = render(<ActivityView {...props} items={[item]} />);
  const list = screen.getByRole("region", { name: "实时日志列表" });
  list.scrollTop = 100;
  fireEvent.scroll(list);
  view.rerender(<ActivityView {...props} items={[{ ...item, key: "new", title: "新记录" }, item]} />);
  expect(screen.queryByText("新记录", { exact: true })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "恢复实时 · 1 条新记录" }));
  expect(list.scrollTop).toBe(0);
  expect(screen.getByText("新记录", { exact: true })).toBeVisible();
});

it.each(["clipboard", "blocked", "http"])("copies diagnostic text or offers selected text when %s is available", async (mode) => {
  const previous = Object.getOwnPropertyDescriptor(navigator, "clipboard");
  const writeText = vi.fn(mode === "blocked" ? () => Promise.reject(new Error("denied")) : () => Promise.resolve());
  Object.defineProperty(navigator, "clipboard", { configurable: true, value: mode === "http" ? undefined : { writeText } });
  try {
    render(<ActivityView items={[{ key: "record", title: "保存失败", detail: "设备不可达", requestId: "trace-123", technicalDetail: "connection refused", tone: "error" }]}
      onClear={vi.fn()} onOpenDetails={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: "复制日志：保存失败" }));
    if (mode === "clipboard") {
      await waitFor(() => expect(writeText).toHaveBeenCalledWith(expect.stringContaining("排查编号：trace-123\nconnection refused")));
      expect(screen.queryByRole("textbox", { name: "手动复制日志：保存失败" })).not.toBeInTheDocument();
    } else {
      const text = await screen.findByRole("textbox", { name: "手动复制日志：保存失败" }) as HTMLTextAreaElement;
      await waitFor(() => expect(text).toHaveFocus());
      expect(text.value).toContain("trace-123");
      expect(text.selectionEnd - text.selectionStart).toBe(text.value.length);
    }
    expect(screen.getByRole("button", { name: "恢复实时" })).toBeVisible();
  } finally {
    if (previous) Object.defineProperty(navigator, "clipboard", previous);
    else Reflect.deleteProperty(navigator, "clipboard");
  }
});
