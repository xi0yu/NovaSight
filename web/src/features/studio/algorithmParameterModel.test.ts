import { expect, it } from "vitest";

import { validateStudioConfigSchema } from "./algorithmParameterModel";
import type { ConfigSchemaResponse } from "../../api";

it("rejects missing, mistyped and unsupported fields in a received parameter schema", () => {
  // Deliberately incomplete input exercises validation, not backend compatibility.
  const schema: ConfigSchemaResponse = {
    version: 1,
    algorithm: {
      id: "continuous_atan_medoid_v2", label: "连续 Atan 控制",
      response: { formula: "", atan_scale_counts: 256 },
      prediction: { model: "", aim_history_points: 4, velocity_segments: 3 }
    },
    values: {},
    sections: [{ id: "pipeline", label: "控制参数", fields: [
      { path: "pipeline.target_class_filter", label: "类别", type: "float", apply_mode: "hot_update", restart_required: false },
      { path: "pipeline.prediction_enabled", label: "预测", type: "float", apply_mode: "epoch_reload", restart_required: false },
      { path: "pipeline.new_backend_parameter", label: "未知参数", type: "float", apply_mode: "hot_update", restart_required: false }
    ] }]
  };
  expect(validateStudioConfigSchema(schema)).toEqual(expect.arrayContaining([
    { path: "pipeline.p_response_scale", reason: "Studio control parameter is not exposed by backend schema" },
    { path: "pipeline.target_class_weights", reason: "Studio targeting parameter is not exposed by backend schema" },
    { path: "pipeline.target_class_filter", reason: "expected string schema field, got float" },
    { path: "pipeline.prediction_enabled", reason: "expected bool schema field, got float" },
    { path: "pipeline.prediction_enabled", reason: "control algorithm field must be hot-applied by the backend" },
    { path: "pipeline.new_backend_parameter", reason: "backend pipeline parameter has no Studio control contract" }
  ]));
});
