import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { StartupResponsePreview, SingleOutputPreview } from "./AlgorithmResponsePreview";

it("draws waiting before smooth ramp-up and updates both durations including zero", () => {
  const view = render(<StartupResponsePreview delayMs={200} rampMs={400} />);
  const chart = screen.getByRole("img", { name: /等待 200 ms，渐增 400 ms，600 ms 后/ });
  const first = chart.querySelector(".response-curve")!.getAttribute("d");
  expect(first).toMatch(/^M36,112 L94.4,112/);
  expect(first).toContain("152.8,72");
  view.rerender(<StartupResponsePreview delayMs={500} rampMs={1000} />);
  expect(screen.getByRole("img", { name: /等待 500 ms，渐增 1000 ms，1500 ms 后/ }).querySelector(".response-curve")!.getAttribute("d")).not.toBe(first);
  view.rerender(<StartupResponsePreview delayMs={200} rampMs={0} />);
  expect(screen.getByRole("img").querySelector(".response-curve")).toHaveAttribute("d", "M36 112H94.4V32H328");
  view.rerender(<StartupResponsePreview delayMs={0} rampMs={0} />);
  expect(screen.getByRole("img").querySelector(".response-curve")).toHaveAttribute("d", "M36 112H36V32H328");
});

it("shows Kp demand and signed axis caps, with zero gain or zero limit suppressing output", () => {
  const view = render(<SingleOutputPreview kp={1} limitX={7} limitY={0} />);
  expect(screen.getByRole("list", { name: "单次输出计算流程" })).toHaveTextContent("7.00 / 0.00");
  const chart = screen.getByRole("img", { name: /Kp 1，X 上限 7，Y 上限 0/ });
  const xs = chart.querySelector(".response-x")!.getAttribute("points")!.split(" ").map((p) => Number(p.split(",")[1]));
  const ys = chart.querySelector(".response-y")!.getAttribute("points")!.split(" ").map((p) => Number(p.split(",")[1]));
  expect(xs[0]).toBeGreaterThan(88);
  expect(xs[xs.length - 1]).toBeLessThan(88);
  expect(new Set(ys)).toEqual(new Set([88]));
  view.rerender(<SingleOutputPreview kp={0} limitX={1000} limitY={1000} />);
  expect(screen.getByRole("list", { name: "单次输出计算流程" })).toHaveTextContent("0.00 / 0.00");
});

it("highlights only capped segments as gain and axis limits change, including a disabled axis", () => {
  const view = render(<SingleOutputPreview kp={0.024} limitX={8} limitY={8} />);
  const marked = (axis: string) => screen.getByRole("img").querySelector(`path[aria-label="${axis} 轴受上限限制"]`);
  expect(marked("X")).toBeNull();
  expect(marked("Y")).toBeNull();
  expect(screen.getByText("琥珀色段：受上限限制")).toBeVisible();

  view.rerender(<SingleOutputPreview kp={0.03} limitX={8} limitY={0} />);
  const segments = marked("X")!.getAttribute("d")!.match(/-?\d+(?:\.\d+)?/g)!.map(Number);
  // Both tails are capped; the central response remains unmarked.
  expect(segments[0]).toBe(36);
  expect(segments[2]).toBeGreaterThan(36);
  expect(segments[2]).toBeLessThan(182);
  expect(segments[3]).toBeGreaterThan(182);
  expect(segments[3]).toBeLessThan(328);
  expect(segments[2] + segments[3]).toBeCloseTo(364);
  expect(segments[1]).toBeGreaterThan(88);
  expect(segments[4]).toBeLessThan(88);
  expect(segments[5]).toBe(328);
  expect(marked("Y")).toHaveAttribute("d", "M36 88H182M182 88H328");

  view.rerender(<SingleOutputPreview kp={0.03} limitX={1000} limitY={0} />);
  expect(marked("X")).toBeNull();
  expect(marked("Y")).not.toBeNull();
  view.rerender(<SingleOutputPreview kp={0} limitX={0} limitY={0} />);
  expect(marked("X")).toBeNull();
  expect(marked("Y")).toBeNull();
});

it("keeps saturated output heights and axis scale fixed while higher Kp reaches the cap sooner", () => {
  const view = render(<SingleOutputPreview kp={0.03} limitX={8} limitY={4} />);
  const chart = () => screen.getByRole("img");
  const endpoints = (axis: string) => {
    const points = chart().querySelector(`.response-${axis}`)!.getAttribute("points")!.split(" ");
    return [points[0], points[points.length - 1]];
  };
  const xs = endpoints("x");
  const ys = endpoints("y");
  const initialBoundary = Number(chart().querySelector(".response-limited")!.getAttribute("d")!.split("M")[2].split(" ")[0]);
  for (const kp of [0.2, 1, 1000]) {
    view.rerender(<SingleOutputPreview kp={kp} limitX={8} limitY={4} />);
    expect(chart()).toHaveAccessibleName(new RegExp("纵轴范围 ±9 counts"));
    expect(endpoints("x")).toEqual(xs);
    expect(endpoints("y")).toEqual(ys);
    const boundary = Number(chart().querySelector(".response-limited")!.getAttribute("d")!.split("M")[2].split(" ")[0]);
    expect(boundary).toBeLessThan(initialBoundary);
    expect(boundary).toBeGreaterThanOrEqual(182);
    // Demand continues beyond the scale; cropping must not turn it into another capped line.
    const demand = chart().querySelector(".response-raw")!;
    expect(Number(demand.getAttribute("points")!.split(" ")[0].split(",")[1])).toBeGreaterThan(144);
    const clipId = chart().querySelector("clipPath")!.id;
    expect(demand.parentElement).toHaveAttribute("clip-path", `url(#${clipId})`);
  }
  view.rerender(<SingleOutputPreview kp={1000} limitX={0} limitY={0} />);
  expect(chart()).toHaveAccessibleName(/纵轴范围 ±1 counts/);
  expect(new Set(chart().querySelector(".response-x")!.getAttribute("points")!.split(" ").map((p) => p.split(",")[1]))).toEqual(new Set(["88"]));
});
