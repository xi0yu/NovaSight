export type ModelProfileStatus =
  | "UNINSPECTED"
  | "INSPECTING"
  | "NEEDS_CONFIGURATION"
  | "READY_FOR_PROBE"
  | "PROBING"
  | "VALIDATED"
  | "INVALID"
  | "INCOMPATIBLE"
  | "ACTIVE";

export interface ModelTensorInspection {
  name: string;
  io_mode: string;
  engine_shape: number[];
  data_type: string;
  tensor_format: null;
  is_shape_tensor: null;
  bytes_per_component: number;
  components_per_element: null;
  vectorized_dim: null;
}

export interface ModelValidationEnvironment {
  schema_version: number;
  novasight_model_abi: number;
  runtime_abi_version: number;
  tensorrt_runtime_version: number;
  cuda_runtime_version: number;
  cuda_driver_version: number;
  device_ordinal: number;
  compute_capability_major: number;
  compute_capability_minor: number;
  integrated: boolean;
  total_global_memory: number;
  device_name: string;
}

export interface ModelProfile {
  schema_version: number;
  model_id: string;
  display_name: string;
  status: ModelProfileStatus;
  engine: {
    path: string;
    sha256: string;
    file_size: number;
    /** Decimal u128 text; JSON numbers cannot preserve Unix nanoseconds. */
    modified_at_ns: string;
  };
  inspection: {
    deserialize_ok: boolean;
    compatible: boolean;
    engine_name: null;
    inputs: ModelTensorInspection[];
    outputs: ModelTensorInspection[];
    profiles: null;
    selected_profile: number;
    has_dynamic_shape: boolean;
    has_shape_input: null;
    requires_plugin: null;
    error_code: null;
    raw_error: string;
    warnings: string[];
  };
  input: {
    name: string;
    runtime_shape: number[];
    engine_shape: number[];
    dtype: string;
    layout: string;
    profile_index: number;
  };
  outputs: Array<{
    name: string;
    shape: number[];
    engine_shape: number[];
    dtype: string;
  }>;
  preprocess: {
    color_format: string;
    scale: number | null;
    offsets: number[];
    mean: number[];
    std: number[];
    resize_mode: string;
    symmetric_padding: boolean;
    padding_value: number;
  };
  decoder: {
    parser_type: string;
    class_count: number;
    bbox_format: string;
    has_objectness: boolean | null;
  };
  postprocess: {
    confidence_threshold: number;
    nms_threshold: number;
    max_detections: number;
  };
  labels: string[];
  parser_candidates: Array<{
    parser_type: string;
    confidence: string;
    reason: string;
    requires_confirmation: boolean;
  }>;
  validation: {
    status: string;
    validated_at: string;
    engine_execution_ok: boolean;
    decoder_ok: boolean;
    nms_ok: boolean;
    detection_batch_ok: boolean;
    profile_fingerprint: string;
    environment?: ModelValidationEnvironment;
    issues: string[];
  };
}

export interface ModelProbeIssue {
  code: string;
  stage: string;
  message: string;
}

export interface ModelProbeReport {
  status: string;
  engine_execution_ok: boolean;
  output_tensor_ok: boolean;
  decoder_ok: boolean;
  nms_ok: boolean;
  detection_batch_ok: boolean;
  preprocess_ms: number | null;
  inference_ms: number;
  decode_ms: number;
  nms_ms: number | null;
  decode_includes_nms: boolean;
  probe_input: string;
  detection_count: number;
  issues: ModelProbeIssue[];
}

export interface ModelProfileResponse {
  artifact_id: number;
  profile_path: string;
  profile: ModelProfile;
  report?: ModelProbeReport;
}

export interface ModelProbeResponse extends ModelProfileResponse {
  report: ModelProbeReport;
}

export interface ModelProfileConfigurePayload {
  color_format: "RGB" | "BGR" | "GRAY";
  scale: number;
  offsets?: number[];
  mean?: number[];
  std?: number[];
  resize_mode: "direct" | "letterbox";
  symmetric_padding?: boolean;
  padding_value?: number;
  parser_type: string;
  class_count: number;
  labels: string[];
  bbox_format: "xywh" | "xyxy";
  has_objectness: boolean;
  confidence_threshold?: number;
  nms_threshold?: number;
  max_detections?: number;
}

export interface DeepStreamManifestRecommendation {
  model_id: string;
  display_name: string;
  runtime_precision: string;
  input_name: string;
  input_shape: number[];
  input_dtype: string;
  input_color_format: string;
  input_scale_factor: number;
  maintain_aspect_ratio: boolean;
  symmetric_padding: boolean;
  output_name: string;
  output_shape: number[];
  output_dtype: string;
  class_count: number;
  confidence_threshold: number;
  nms_iou_threshold: number;
}

export interface DeepStreamRecommendationResponse {
  artifact_id: number;
  artifact_path: string;
  recommendation: DeepStreamManifestRecommendation;
  io_tensors: Array<{
    name: string;
    shape: number[];
    dtype: string;
    mode: "input" | "output";
  }>;
  class_names: string[];
  output_has_objectness: boolean;
  sources: Record<string, string>;
  warnings: string[];
}

type JsonRecord = Record<string, unknown>;

export class ModelIngressContractError extends Error {
  constructor(path: string, expected: string) {
    super(`模型检查数据契约错误：${path} 应为 ${expected}`);
    this.name = "ModelIngressContractError";
  }
}

function expectRecord(value: unknown, path: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ModelIngressContractError(path, "object");
  }
  return value as JsonRecord;
}

function expectString(value: unknown, path: string): string {
  if (typeof value !== "string") throw new ModelIngressContractError(path, "string");
  return value;
}

function expectBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new ModelIngressContractError(path, "boolean");
  return value;
}

function expectFiniteNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ModelIngressContractError(path, "finite number");
  }
  return value;
}

function expectInteger(value: unknown, path: string): number {
  const number = expectFiniteNumber(value, path);
  if (!Number.isSafeInteger(number)) {
    throw new ModelIngressContractError(path, "safe integer");
  }
  return number;
}

function expectUnsignedInteger(value: unknown, path: string): number {
  const number = expectInteger(value, path);
  if (number < 0) throw new ModelIngressContractError(path, "unsigned safe integer");
  return number;
}

function expectNullableNumber(value: unknown, path: string): number | null {
  return value === null ? null : expectFiniteNumber(value, path);
}

function expectNull(value: unknown, path: string): null {
  if (value !== null) throw new ModelIngressContractError(path, "null");
  return null;
}

function expectArray<T>(
  value: unknown,
  path: string,
  decode: (item: unknown, itemPath: string) => T
): T[] {
  if (!Array.isArray(value)) throw new ModelIngressContractError(path, "array");
  return value.map((item, index) => decode(item, `${path}[${index}]`));
}

function expectStringArray(value: unknown, path: string): string[] {
  return expectArray(value, path, expectString);
}

function expectNumberArray(value: unknown, path: string): number[] {
  return expectArray(value, path, expectFiniteNumber);
}

function expectUnsignedIntegerArray(value: unknown, path: string): number[] {
  return expectArray(value, path, expectUnsignedInteger);
}

function decodeTensorInspection(value: unknown, path: string): ModelTensorInspection {
  const record = expectRecord(value, path);
  return {
    name: expectString(record.name, `${path}.name`),
    io_mode: expectString(record.io_mode, `${path}.io_mode`),
    engine_shape: expectUnsignedIntegerArray(record.engine_shape, `${path}.engine_shape`),
    data_type: expectString(record.data_type, `${path}.data_type`),
    tensor_format: expectNull(record.tensor_format, `${path}.tensor_format`),
    is_shape_tensor: expectNull(record.is_shape_tensor, `${path}.is_shape_tensor`),
    bytes_per_component: expectUnsignedInteger(
      record.bytes_per_component,
      `${path}.bytes_per_component`
    ),
    components_per_element: expectNull(
      record.components_per_element,
      `${path}.components_per_element`
    ),
    vectorized_dim: expectNull(record.vectorized_dim, `${path}.vectorized_dim`)
  };
}

function decodeValidationEnvironment(
  value: unknown,
  path: string
): ModelValidationEnvironment {
  const record = expectRecord(value, path);
  return {
    schema_version: expectUnsignedInteger(record.schema_version, `${path}.schema_version`),
    novasight_model_abi: expectUnsignedInteger(
      record.novasight_model_abi,
      `${path}.novasight_model_abi`
    ),
    runtime_abi_version: expectUnsignedInteger(
      record.runtime_abi_version,
      `${path}.runtime_abi_version`
    ),
    tensorrt_runtime_version: expectInteger(
      record.tensorrt_runtime_version,
      `${path}.tensorrt_runtime_version`
    ),
    cuda_runtime_version: expectInteger(
      record.cuda_runtime_version,
      `${path}.cuda_runtime_version`
    ),
    cuda_driver_version: expectInteger(
      record.cuda_driver_version,
      `${path}.cuda_driver_version`
    ),
    device_ordinal: expectInteger(record.device_ordinal, `${path}.device_ordinal`),
    compute_capability_major: expectInteger(
      record.compute_capability_major,
      `${path}.compute_capability_major`
    ),
    compute_capability_minor: expectInteger(
      record.compute_capability_minor,
      `${path}.compute_capability_minor`
    ),
    integrated: expectBoolean(record.integrated, `${path}.integrated`),
    total_global_memory: expectUnsignedInteger(
      record.total_global_memory,
      `${path}.total_global_memory`
    ),
    device_name: expectString(record.device_name, `${path}.device_name`)
  };
}

function decodeModelProfile(value: unknown, path: string): ModelProfile {
  const record = expectRecord(value, path);
  const status = record.status;
  const statuses: readonly ModelProfileStatus[] = [
    "UNINSPECTED",
    "INSPECTING",
    "NEEDS_CONFIGURATION",
    "READY_FOR_PROBE",
    "PROBING",
    "VALIDATED",
    "INVALID",
    "INCOMPATIBLE",
    "ACTIVE"
  ];
  if (typeof status !== "string" || !(statuses as readonly string[]).includes(status)) {
    throw new ModelIngressContractError(`${path}.status`, statuses.join(" | "));
  }
  const engine = expectRecord(record.engine, `${path}.engine`);
  const inspection = expectRecord(record.inspection, `${path}.inspection`);
  const input = expectRecord(record.input, `${path}.input`);
  const preprocess = expectRecord(record.preprocess, `${path}.preprocess`);
  const decoder = expectRecord(record.decoder, `${path}.decoder`);
  const postprocess = expectRecord(record.postprocess, `${path}.postprocess`);
  const validation = expectRecord(record.validation, `${path}.validation`);
  const environment = "environment" in validation
    ? decodeValidationEnvironment(validation.environment, `${path}.validation.environment`)
    : undefined;
  return {
    schema_version: expectUnsignedInteger(record.schema_version, `${path}.schema_version`),
    model_id: expectString(record.model_id, `${path}.model_id`),
    display_name: expectString(record.display_name, `${path}.display_name`),
    status: status as ModelProfileStatus,
    engine: {
      path: expectString(engine.path, `${path}.engine.path`),
      sha256: expectString(engine.sha256, `${path}.engine.sha256`),
      file_size: expectUnsignedInteger(engine.file_size, `${path}.engine.file_size`),
      modified_at_ns: expectString(engine.modified_at_ns, `${path}.engine.modified_at_ns`)
    },
    inspection: {
      deserialize_ok: expectBoolean(
        inspection.deserialize_ok,
        `${path}.inspection.deserialize_ok`
      ),
      compatible: expectBoolean(inspection.compatible, `${path}.inspection.compatible`),
      engine_name: expectNull(inspection.engine_name, `${path}.inspection.engine_name`),
      inputs: expectArray(
        inspection.inputs,
        `${path}.inspection.inputs`,
        decodeTensorInspection
      ),
      outputs: expectArray(
        inspection.outputs,
        `${path}.inspection.outputs`,
        decodeTensorInspection
      ),
      profiles: expectNull(inspection.profiles, `${path}.inspection.profiles`),
      selected_profile: expectUnsignedInteger(
        inspection.selected_profile,
        `${path}.inspection.selected_profile`
      ),
      has_dynamic_shape: expectBoolean(
        inspection.has_dynamic_shape,
        `${path}.inspection.has_dynamic_shape`
      ),
      has_shape_input: expectNull(
        inspection.has_shape_input,
        `${path}.inspection.has_shape_input`
      ),
      requires_plugin: expectNull(
        inspection.requires_plugin,
        `${path}.inspection.requires_plugin`
      ),
      error_code: expectNull(inspection.error_code, `${path}.inspection.error_code`),
      raw_error: expectString(inspection.raw_error, `${path}.inspection.raw_error`),
      warnings: expectStringArray(inspection.warnings, `${path}.inspection.warnings`)
    },
    input: {
      name: expectString(input.name, `${path}.input.name`),
      runtime_shape: expectUnsignedIntegerArray(
        input.runtime_shape,
        `${path}.input.runtime_shape`
      ),
      engine_shape: expectUnsignedIntegerArray(input.engine_shape, `${path}.input.engine_shape`),
      dtype: expectString(input.dtype, `${path}.input.dtype`),
      layout: expectString(input.layout, `${path}.input.layout`),
      profile_index: expectUnsignedInteger(input.profile_index, `${path}.input.profile_index`)
    },
    outputs: expectArray(record.outputs, `${path}.outputs`, (output, outputPath) => {
      const outputRecord = expectRecord(output, outputPath);
      return {
        name: expectString(outputRecord.name, `${outputPath}.name`),
        shape: expectUnsignedIntegerArray(outputRecord.shape, `${outputPath}.shape`),
        engine_shape: expectUnsignedIntegerArray(
          outputRecord.engine_shape,
          `${outputPath}.engine_shape`
        ),
        dtype: expectString(outputRecord.dtype, `${outputPath}.dtype`)
      };
    }),
    preprocess: {
      color_format: expectString(preprocess.color_format, `${path}.preprocess.color_format`),
      scale: expectNullableNumber(preprocess.scale, `${path}.preprocess.scale`),
      offsets: expectNumberArray(preprocess.offsets, `${path}.preprocess.offsets`),
      mean: expectNumberArray(preprocess.mean, `${path}.preprocess.mean`),
      std: expectNumberArray(preprocess.std, `${path}.preprocess.std`),
      resize_mode: expectString(preprocess.resize_mode, `${path}.preprocess.resize_mode`),
      symmetric_padding: expectBoolean(
        preprocess.symmetric_padding,
        `${path}.preprocess.symmetric_padding`
      ),
      padding_value: expectFiniteNumber(
        preprocess.padding_value,
        `${path}.preprocess.padding_value`
      )
    },
    decoder: {
      parser_type: expectString(decoder.parser_type, `${path}.decoder.parser_type`),
      class_count: expectUnsignedInteger(decoder.class_count, `${path}.decoder.class_count`),
      bbox_format: expectString(decoder.bbox_format, `${path}.decoder.bbox_format`),
      has_objectness: decoder.has_objectness === null
        ? null
        : expectBoolean(decoder.has_objectness, `${path}.decoder.has_objectness`)
    },
    postprocess: {
      confidence_threshold: expectFiniteNumber(
        postprocess.confidence_threshold,
        `${path}.postprocess.confidence_threshold`
      ),
      nms_threshold: expectFiniteNumber(
        postprocess.nms_threshold,
        `${path}.postprocess.nms_threshold`
      ),
      max_detections: expectUnsignedInteger(
        postprocess.max_detections,
        `${path}.postprocess.max_detections`
      )
    },
    labels: expectStringArray(record.labels, `${path}.labels`),
    parser_candidates: expectArray(
      record.parser_candidates,
      `${path}.parser_candidates`,
      (candidate, candidatePath) => {
        const candidateRecord = expectRecord(candidate, candidatePath);
        return {
          parser_type: expectString(candidateRecord.parser_type, `${candidatePath}.parser_type`),
          confidence: expectString(candidateRecord.confidence, `${candidatePath}.confidence`),
          reason: expectString(candidateRecord.reason, `${candidatePath}.reason`),
          requires_confirmation: expectBoolean(
            candidateRecord.requires_confirmation,
            `${candidatePath}.requires_confirmation`
          )
        };
      }
    ),
    validation: {
      status: expectString(validation.status, `${path}.validation.status`),
      validated_at: expectString(validation.validated_at, `${path}.validation.validated_at`),
      engine_execution_ok: expectBoolean(
        validation.engine_execution_ok,
        `${path}.validation.engine_execution_ok`
      ),
      decoder_ok: expectBoolean(validation.decoder_ok, `${path}.validation.decoder_ok`),
      nms_ok: expectBoolean(validation.nms_ok, `${path}.validation.nms_ok`),
      detection_batch_ok: expectBoolean(
        validation.detection_batch_ok,
        `${path}.validation.detection_batch_ok`
      ),
      profile_fingerprint: expectString(
        validation.profile_fingerprint,
        `${path}.validation.profile_fingerprint`
      ),
      ...(environment ? { environment } : {}),
      issues: expectStringArray(validation.issues, `${path}.validation.issues`)
    }
  };
}

function decodeProbeIssue(value: unknown, path: string): ModelProbeIssue {
  const record = expectRecord(value, path);
  return {
    code: expectString(record.code, `${path}.code`),
    stage: expectString(record.stage, `${path}.stage`),
    message: expectString(record.message, `${path}.message`)
  };
}

function decodeProbeReport(value: unknown, path: string): ModelProbeReport {
  const record = expectRecord(value, path);
  return {
    status: expectString(record.status, `${path}.status`),
    engine_execution_ok: expectBoolean(
      record.engine_execution_ok,
      `${path}.engine_execution_ok`
    ),
    output_tensor_ok: expectBoolean(record.output_tensor_ok, `${path}.output_tensor_ok`),
    decoder_ok: expectBoolean(record.decoder_ok, `${path}.decoder_ok`),
    nms_ok: expectBoolean(record.nms_ok, `${path}.nms_ok`),
    detection_batch_ok: expectBoolean(
      record.detection_batch_ok,
      `${path}.detection_batch_ok`
    ),
    preprocess_ms: expectNullableNumber(record.preprocess_ms, `${path}.preprocess_ms`),
    inference_ms: expectFiniteNumber(record.inference_ms, `${path}.inference_ms`),
    decode_ms: expectFiniteNumber(record.decode_ms, `${path}.decode_ms`),
    nms_ms: expectNullableNumber(record.nms_ms, `${path}.nms_ms`),
    decode_includes_nms: expectBoolean(
      record.decode_includes_nms,
      `${path}.decode_includes_nms`
    ),
    probe_input: expectString(record.probe_input, `${path}.probe_input`),
    detection_count: expectUnsignedInteger(record.detection_count, `${path}.detection_count`),
    issues: expectArray(record.issues, `${path}.issues`, decodeProbeIssue)
  };
}

function decodeProfileResponseAt(value: unknown, requireReport: boolean): ModelProfileResponse {
  const path = requireReport ? "model_probe" : "model_profile";
  const record = expectRecord(value, path);
  const report = "report" in record
    ? decodeProbeReport(record.report, `${path}.report`)
    : undefined;
  if (requireReport && !report) {
    throw new ModelIngressContractError(`${path}.report`, "object");
  }
  return {
    artifact_id: expectInteger(record.artifact_id, `${path}.artifact_id`),
    profile_path: expectString(record.profile_path, `${path}.profile_path`),
    profile: decodeModelProfile(record.profile, `${path}.profile`),
    ...(report ? { report } : {})
  };
}

export function decodeModelProfileResponse(value: unknown): ModelProfileResponse {
  return decodeProfileResponseAt(value, false);
}

export function decodeModelProbeResponse(value: unknown): ModelProbeResponse {
  return decodeProfileResponseAt(value, true) as ModelProbeResponse;
}

export function decodeDeepStreamRecommendation(
  value: unknown
): DeepStreamRecommendationResponse {
  const path = "deepstream_recommendation";
  const record = expectRecord(value, path);
  const recommendation = expectRecord(record.recommendation, `${path}.recommendation`);
  const sources = expectRecord(record.sources, `${path}.sources`);
  const decodedSources: Record<string, string> = {};
  for (const [key, source] of Object.entries(sources)) {
    decodedSources[key] = expectString(source, `${path}.sources.${key}`);
  }
  return {
    artifact_id: expectInteger(record.artifact_id, `${path}.artifact_id`),
    artifact_path: expectString(record.artifact_path, `${path}.artifact_path`),
    recommendation: {
      model_id: expectString(recommendation.model_id, `${path}.recommendation.model_id`),
      display_name: expectString(
        recommendation.display_name,
        `${path}.recommendation.display_name`
      ),
      runtime_precision: expectString(
        recommendation.runtime_precision,
        `${path}.recommendation.runtime_precision`
      ),
      input_name: expectString(recommendation.input_name, `${path}.recommendation.input_name`),
      input_shape: expectUnsignedIntegerArray(
        recommendation.input_shape,
        `${path}.recommendation.input_shape`
      ),
      input_dtype: expectString(recommendation.input_dtype, `${path}.recommendation.input_dtype`),
      input_color_format: expectString(
        recommendation.input_color_format,
        `${path}.recommendation.input_color_format`
      ),
      input_scale_factor: expectFiniteNumber(
        recommendation.input_scale_factor,
        `${path}.recommendation.input_scale_factor`
      ),
      maintain_aspect_ratio: expectBoolean(
        recommendation.maintain_aspect_ratio,
        `${path}.recommendation.maintain_aspect_ratio`
      ),
      symmetric_padding: expectBoolean(
        recommendation.symmetric_padding,
        `${path}.recommendation.symmetric_padding`
      ),
      output_name: expectString(
        recommendation.output_name,
        `${path}.recommendation.output_name`
      ),
      output_shape: expectUnsignedIntegerArray(
        recommendation.output_shape,
        `${path}.recommendation.output_shape`
      ),
      output_dtype: expectString(
        recommendation.output_dtype,
        `${path}.recommendation.output_dtype`
      ),
      class_count: expectUnsignedInteger(
        recommendation.class_count,
        `${path}.recommendation.class_count`
      ),
      confidence_threshold: expectFiniteNumber(
        recommendation.confidence_threshold,
        `${path}.recommendation.confidence_threshold`
      ),
      nms_iou_threshold: expectFiniteNumber(
        recommendation.nms_iou_threshold,
        `${path}.recommendation.nms_iou_threshold`
      )
    },
    io_tensors: expectArray(record.io_tensors, `${path}.io_tensors`, (tensor, tensorPath) => {
      const tensorRecord = expectRecord(tensor, tensorPath);
      const mode = tensorRecord.mode;
      if (mode !== "input" && mode !== "output") {
        throw new ModelIngressContractError(`${tensorPath}.mode`, "input | output");
      }
      return {
        name: expectString(tensorRecord.name, `${tensorPath}.name`),
        shape: expectUnsignedIntegerArray(tensorRecord.shape, `${tensorPath}.shape`),
        dtype: expectString(tensorRecord.dtype, `${tensorPath}.dtype`),
        mode
      };
    }),
    class_names: expectStringArray(record.class_names, `${path}.class_names`),
    output_has_objectness: expectBoolean(
      record.output_has_objectness,
      `${path}.output_has_objectness`
    ),
    sources: decodedSources,
    warnings: expectStringArray(record.warnings, `${path}.warnings`)
  };
}
