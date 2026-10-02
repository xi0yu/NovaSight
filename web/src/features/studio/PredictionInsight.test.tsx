import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PredictionInsight, predictionInsight, ResponseExperiment, responseIllustration } from "./PredictionInsight";

it("lets users inspect draft gain, time normalization and zero/signed demand without sending commands", () => {
  const { rerender } = render(<ResponseExperiment kp={0.2} rampMs={200} referenceHz={60} />);
  const initial = Number(screen.getByLabelText("试算输出").textContent);
  fireEvent.change(screen.getByLabelText("模拟控制频率"), { target: { value: "120" } });
  expect(Number(screen.getByLabelText("试算输出").textContent)).toBeCloseTo(initial / 2, 1);
  fireEvent.change(screen.getByLabelText("试算入场时间"), { target: { value: "0" } });
  expect(screen.getByLabelText("试算输出").textContent).toBe("0.00");
  fireEvent.change(screen.getByLabelText("试算入场时间"), { target: { value: "200" } });
  expect(screen.getByRole("img", { name: /当前达到设定 Kp 的 100%/ })).toBeTruthy();
  rerender(<ResponseExperiment kp={0} rampMs={0} referenceHz={0} />);
  expect(screen.getByLabelText("试算输出").textContent).toBe("0.00");
  expect(responseIllustration(0.2, 0, 0, 0, 60, -100).demand).toBeLessThan(0);
  expect(responseIllustration(0.2, 0, 60, 0, 60, 100).demand).toBe(0);
});

const sample = {
  prediction_allowed: true, prediction_allowed_y: true,
  prediction_safe_offset_x: 3, prediction_safe_offset_y: 4,
  prediction_raw_offset_x: 6, prediction_raw_offset_y: 8,
  prediction_horizon_ms: 40, frame_age_ms: 20,
  prediction_actuation_delay_ms: 4, prediction_lead_ms: 16,
};

describe("prediction contribution", () => {
  it("distinguishes capped prediction from an ineffective switch", () => {
    const view = predictionInsight(sample, true);
    expect(view.status).toBe("已到位移上限");
    expect(view.offset).toBe(5);
    expect(view.raw).toBe(10);
    expect(view.requested).toBe(40);
    render(<PredictionInsight sample={sample} live configuredEnabled />);
    expect(screen.getByRole("img", { name: /预测位移 5.0/ })).toBeTruthy();
    expect(screen.getByText("5.0 px")).toBeTruthy();
  });
  it("does not show old measurements as current when stopped or disconnected", () => {
    const view = predictionInsight(sample, false);
    expect(view.status).toBe("等待实时观测");
    expect(view.offset).toBeNull();
    expect(view.horizon).toBeNull();
  });
  it("explains old backend time clipping and reversal suppression separately", () => {
    expect(predictionInsight({ ...sample, prediction_horizon_ms: 12.5 }, true).status).toBe("运行时域被缩短");
    expect(predictionInsight({ ...sample, motion_state: "unstable", prediction_allowed: false }, true).status).toBe("方向变化 · 暂停外推");
  });
  it("does not confuse missing or nonfinite offsets with zero motion", () => {
    expect(predictionInsight({ prediction_allowed: false }, true).status).toBe("预测尚未生效");
    expect(predictionInsight({ ...sample, prediction_safe_offset_x: NaN }, true).offset).toBeNull();
    const view = predictionInsight({ ...sample, prediction_safe_offset_x: 0, prediction_safe_offset_y: 0, prediction_raw_offset_x: 0, prediction_raw_offset_y: 0 }, true);
    expect(view.status).toBe("预测偏移很小");
  });
});
