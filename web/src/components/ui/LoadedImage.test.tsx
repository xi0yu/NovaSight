import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LoadedImage } from "./LoadedImage";

describe("complete image presentation", () => {
  beforeEach(() => {
    vi.spyOn(HTMLImageElement.prototype, "complete", "get").mockReturnValue(false);
    vi.spyOn(HTMLImageElement.prototype, "naturalWidth", "get").mockReturnValue(320);
  });

  it.each(["download", "cache"])("keeps the placeholder until the %s image has decoded", async (source) => {
    let finish!: () => void;
    const decoded = new Promise<void>((resolve) => { finish = resolve; });
    Object.defineProperty(HTMLImageElement.prototype, "decode", { configurable: true, value: vi.fn(() => decoded) });
    if (source === "cache") vi.spyOn(HTMLImageElement.prototype, "complete", "get").mockReturnValue(true);
    render(<LoadedImage src="/device.png" alt="设备照片" width={320} height={320} loading="lazy" />);
    const image = screen.getByAltText("设备照片");
    expect(image).toHaveAttribute("loading", "lazy");
    if (source === "download") fireEvent.load(image);
    expect(screen.getByRole("img", { name: "设备照片，加载中" })).toHaveAttribute("aria-busy", "true");
    await act(async () => { finish(); await decoded; });
    await waitFor(() => expect(screen.queryByRole("img", { name: "设备照片，加载中" })).not.toBeInTheDocument());
    expect(image.parentElement).toHaveAttribute("aria-busy", "false");
  });

  it.each(["network", "decode"])("shows a stable fallback after a %s failure", async (failure) => {
    Object.defineProperty(HTMLImageElement.prototype, "decode", { configurable: true, value: vi.fn().mockRejectedValue(new Error("invalid image")) });
    render(<LoadedImage src="/broken.png" alt="设备照片" width={320} height={320} />);
    fireEvent(screen.getByAltText("设备照片"), new Event(failure === "network" ? "error" : "load"));
    const fallback = await screen.findByRole("img", { name: "设备照片，图片加载失败" });
    expect(fallback).toHaveAttribute("aria-busy", "false");
    expect(fallback).toHaveAttribute("title", "图片加载失败");
  });

  it("starts a new placeholder on a source change and ignores an obsolete decode", async () => {
    let finishOld!: () => void;
    let finishNew!: () => void;
    const oldDecode = new Promise<void>((resolve) => { finishOld = resolve; });
    const newDecode = new Promise<void>((resolve) => { finishNew = resolve; });
    Object.defineProperty(HTMLImageElement.prototype, "decode", { configurable: true, value: vi.fn(function(this: HTMLImageElement) {
      return this.getAttribute("src") === "/old.png" ? oldDecode : newDecode;
    }) });
    const view = render(<LoadedImage src="/old.png" alt="设备照片" />);
    fireEvent.load(screen.getByAltText("设备照片"));
    view.rerender(<LoadedImage src="/new.png" alt="设备照片" />);
    fireEvent.load(screen.getByAltText("设备照片"));
    await act(async () => { finishOld(); await oldDecode; });
    expect(screen.getByRole("img", { name: "设备照片，加载中" })).toHaveAttribute("aria-busy", "true");
    await act(async () => { finishNew(); await newDecode; });
    expect(screen.queryByRole("img", { name: "设备照片，加载中" })).not.toBeInTheDocument();
    expect(screen.getByAltText("设备照片")).toHaveAttribute("src", "/new.png");
    view.rerender(<LoadedImage src="/third.png" alt="设备照片" />);
    expect(screen.getByRole("img", { name: "设备照片，加载中" })).toHaveAttribute("aria-busy", "true");
  });
});
