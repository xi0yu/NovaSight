import { describe, expect, it } from "vitest";

import { decodeModelPublishResponse } from "./model";

const response = {
  deployment: { id: 1, project_id: 2, artifact_id: 3, previous_artifact_id: null, updated_seq: 4 },
  inference: { selected: "deepstream", available: true, loaded: true, configured: true, reason: "ready" },
  parser_contract: {
    requested_preset: "auto",
    output_family: "yolov8_yolo11",
    has_objectness: false,
    parser_library: "novasight_builtin_cuda",
    parser_function: "YoloGpuPostprocessor",
    nms_owner: "cuda"
  },
  preparation: { manifest_action: "generated", reason: "ready", input_shape: "1x3x256x256", classes: ["enemy"] },
  report: {
    action: "publish", applied: true, rolled_back: false, message: "ready", runtime_error: "",
    artifact_id: 3, previous_artifact_id: null, artifact_path: "data/models/model.engine",
    backend: "deepstream", input_shape: "1x3x256x256", classes: 1, sections: []
  }
};

describe("model publish GPU parser contract", () => {
  it("accepts the CUDA parser and NMS reported by the current daemon", () => {
    expect(decodeModelPublishResponse(response).parser_contract?.nms_owner).toBe("cuda");
  });

  it("does not silently accept the old CPU parser contract", () => {
    expect(() => decodeModelPublishResponse({
      ...response,
      parser_contract: { ...response.parser_contract, parser_library: "novasight_builtin" }
    })).toThrow(/parser_library/);
  });
});
