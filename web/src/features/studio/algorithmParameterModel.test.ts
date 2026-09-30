import { describe, expect, it } from "vitest";

import {
  CONTROL_PIPELINE_FIELDS,
  TARGETING_PIPELINE_FIELDS,
  buildTargetingParameterGroups,
  validateStudioConfigSchema,
  type TargetingParameterValues
} from "./algorithmParameterModel";
import type { ConfigSchemaResponse } from "../../api";

const values: TargetingParameterValues = {
  candidateRatioMaxAspect: 6,
  targetSwitchPreferenceAdvantage: 0.08,
  targetSwitchContinuityScore: 0.7,
  trackerMaxMatchDistance: 1.5,
  trackerPositionCostWeight: 0.75,
  trackerIouCostWeight: 0.25,
  trackerScaleCostWeight: 0.15,
  trackerClassCostWeight: 0.35,
  trackerMaxSizeRatio: 2.5,
  trackerMaxAssociationDtMs: 150,
  trackerKalmanAccelerationNoise: 1200,
  trackerKalmanMeasurementNoiseX: 16,
  trackerKalmanMeasurementNoiseY: 16,
  trackerKalmanMaxPredictDtMs: 35,
  trackerKalmanMaxPredictMissingMs: 80,
  trackerKalmanNisThreshold: 9.21,
  trackerKalmanNisHardReject: 16,
};

describe("target decision parameter model", () => {
  it("checks backend pipeline fields in both directions and verifies targeting types", () => {
    const fields = [...CONTROL_PIPELINE_FIELDS, ...TARGETING_PIPELINE_FIELDS].map((key) => ({
      path: `pipeline.${key}`,
      label: key,
      type: key === "prediction_enabled" || key === "fire_delay_enabled" ? "bool" as const
        : key === "target_class_weights" || key === "target_class_aim_x_ratios" || key === "target_class_filter" || key === "target_class_aim_y_ratios" ? "string" as const
          : "float" as const,
      apply_mode: "hot_update" as const,
      restart_required: false
    }));
    const schema: ConfigSchemaResponse = {
      version: 1,
      algorithm: {
        id: "continuous_atan_medoid_v2", label: "连续 Atan 控制",
        response: { formula: "", atan_scale_counts: 256 },
        prediction: { model: "", aim_history_points: 4, velocity_segments: 3 }
      },
      values: {},
      sections: [{ id: "pipeline", label: "Rust 实时控制", fields }]
    };
    expect(validateStudioConfigSchema(schema)).toEqual([]);
    expect(validateStudioConfigSchema({ ...schema, sections: [{ ...schema.sections[0], fields: [
      ...fields, { ...fields[0], path: "pipeline.new_backend_parameter" }
    ] }] })).toContainEqual({
      path: "pipeline.new_backend_parameter",
      reason: "backend pipeline parameter has no Studio control contract"
    });
    expect(validateStudioConfigSchema({ ...schema, sections: [{ ...schema.sections[0], fields: fields.map((field) => (
      field.path === "pipeline.target_class_filter" ? { ...field, type: "float" as const } : field
    )) }] })).toContainEqual({
      path: "pipeline.target_class_filter", reason: "expected string schema field, got float"
    });
  });

  it("keeps current-evidence scoring and a time-based tracking window", () => {
    for (const key of ["target_selection_size_weight", "target_selection_continuity_weight",
      "target_selection_motion_weight", "target_selection_motion_horizon_ms",
      "tracker_kalman_max_predict_steps", "target_class_priority", "target_range_scale",
      "target_min_confidence", "target_switch_delay_ms", "target_track_max_lost_age_ms",
      "target_selection_distance_weight", "target_selection_class_weight", "target_selection_confidence_weight"]) {
      expect(TARGETING_PIPELINE_FIELDS).not.toContain(key);
    }
    expect(TARGETING_PIPELINE_FIELDS).toContain("tracker_class_cost_weight");
    expect(TARGETING_PIPELINE_FIELDS).not.toContain("target_selection_class_ratio" as never);

    const groups = buildTargetingParameterGroups(values);
    expect(groups.trackerCoreParameters.map(({ key }) => key)).toContain(
      "tracker_class_cost_weight"
    );
  });
});
