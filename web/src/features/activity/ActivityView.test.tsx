import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";

import { ActivityView } from "./ActivityView";

it("separates completed operations from items that need attention", () => {
  const { rerender } = render(<ActivityView items={[{
    key: "success-1",
    title: "模型已更新",
    detail: "新模型已经部署。",
    tone: "success",
    time: Date.now(),
  }]} onOpenDetails={vi.fn()} />);

  expect(screen.getByRole("heading", { name: "最近操作已完成" })).toBeVisible();
  expect(screen.queryByRole("button", { name: "查看完整详情" })).not.toBeInTheDocument();

  rerender(<ActivityView items={[{
    key: "error-1",
    title: "运行故障",
    detail: "推理链已停止。",
    technicalDetail: "backend detail",
    tone: "error",
  }]} onOpenDetails={vi.fn()} />);

  expect(screen.getByRole("heading", { name: "1 件事需要注意" })).toBeVisible();
  expect(screen.getByRole("button", { name: "查看完整详情" })).toBeVisible();
  expect(screen.getByText("原始错误与开发者详情")).toBeVisible();
});
