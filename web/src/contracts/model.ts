export interface ModelProject {
  id: number;
  name: string;
  description: string;
}

export interface ModelVersion {
  id: number;
  project_id: number;
  version: string;
  source_kind: string;
  source_path: string;
  classes: string[];
  input_shape: string;
}

export interface ModelArtifact {
  id: number;
  version_id: number;
  kind: string;
  path: string;
  checksum: string;
  status: string;
  size_bytes: number | null;
}

export interface Deployment {
  id: number;
  project_id: number;
  artifact_id: number;
  previous_artifact_id: number | null;
  updated_seq: number;
}

/** Exact JSON shape of `novasight_store::ActiveModelDeployment`. */
export interface ActiveModel {
  deployment: Deployment;
  project: ModelProject;
  version: ModelVersion;
  artifact: ModelArtifact;
  artifact_path: string;
}

export type ModelRecommendation = "recommended" | "not_recommended" | "unrated";

export interface ModelArtifactMetadata {
  artifact_id: number;
  recommendation: ModelRecommendation;
  tags: string[];
}

export interface ModelCatalogModel {
  type: "model";
  name: string;
  relative_path: string;
  kind: string;
  size_bytes: number;
  scan_status: string;
  scan_reason: string;
  project_id?: number;
  project_name?: string;
  version_id?: number;
  version_name?: string;
  artifact_id?: number;
  artifact_status?: string;
  recommendation: ModelRecommendation;
  tags: string[];
}

export interface ModelCatalogDirectory {
  type: "directory";
  name: string;
  relative_path: string;
  children: Array<ModelCatalogDirectory | ModelCatalogModel>;
}

export interface ModelCatalogResponse {
  root: ModelCatalogDirectory;
  directory_count: number;
  model_count: number;
  discovered_files: number;
  updated_files: number;
  cache_hits: number;
  force: boolean;
}

/** Exact JSON shape of `novasight_store::CatalogEngineRegistration`. */
export interface ModelCatalogRegisterResponse {
  project: ModelProject;
  version: ModelVersion;
  artifact: ModelArtifact;
  engine_path: string;
  created: boolean;
}

export interface ConversionJob {
  id: number;
  version_id: number;
  target_kind: string;
  command: string[];
  status: string;
  log: string;
}

export type ParserPresetId =
  | "auto"
  | "yolov5"
  | "yolov8"
  | "yolo11"
  | "novasight_generic";

export interface ParserContract {
  requested_preset: ParserPresetId;
  output_family: "yolov5" | "yolov8_yolo11";
  has_objectness: boolean;
  parser_library: "novasight_builtin";
  parser_function: "NvDsInferParseNovaSight";
  nms_owner: "deepstream";
}

export interface ModelSwitchInference {
  selected: string;
  available: boolean;
  loaded: boolean;
  configured: boolean;
  reason: string;
}

export interface ModelPreparation {
  manifest_action: "generated" | "reused" | "unchanged";
  reason: string;
  input_shape: string;
  classes: string[];
}

export interface OperationReportSection {
  section: string;
  impact: string;
  status: string;
  message: string;
}

export interface ModelSwitchReport {
  action: string;
  applied: boolean;
  rolled_back: boolean;
  message: string;
  runtime_error: string;
  artifact_id: number;
  previous_artifact_id: number | null;
  artifact_path: string;
  backend: string;
  input_shape: string;
  classes: number;
  sections: OperationReportSection[];
}

export interface ModelPublishResponse {
  deployment: Deployment;
  inference: ModelSwitchInference;
  parser_contract?: ParserContract;
  preparation: ModelPreparation;
  report: ModelSwitchReport;
}

type JsonRecord = Record<string, unknown>;

export class ModelContractError extends Error {
  constructor(path: string, expected: string) {
    super(`模型数据契约错误：${path} 应为 ${expected}`);
    this.name = "ModelContractError";
  }
}

function expectRecord(value: unknown, path: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ModelContractError(path, "object");
  }
  return value as JsonRecord;
}

function expectString(value: unknown, path: string): string {
  if (typeof value !== "string") throw new ModelContractError(path, "string");
  return value;
}

function expectBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new ModelContractError(path, "boolean");
  return value;
}

function expectInteger(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new ModelContractError(path, "safe integer");
  }
  return value;
}

function expectUnsignedInteger(value: unknown, path: string): number {
  const number = expectInteger(value, path);
  if (number < 0) throw new ModelContractError(path, "unsigned safe integer");
  return number;
}

function expectStringArray(value: unknown, path: string): string[] {
  if (!Array.isArray(value)) throw new ModelContractError(path, "string[]");
  return value.map((item, index) => expectString(item, `${path}[${index}]`));
}

function expectRecommendation(value: unknown, path: string): ModelRecommendation {
  if (value === "recommended" || value === "not_recommended" || value === "unrated") {
    return value;
  }
  throw new ModelContractError(path, "recommended | not_recommended | unrated");
}

function expectLiteral<T extends string>(
  value: unknown,
  path: string,
  allowed: readonly T[]
): T {
  if (typeof value === "string" && (allowed as readonly string[]).includes(value)) {
    return value as T;
  }
  throw new ModelContractError(path, allowed.join(" | "));
}

function optionalInteger(record: JsonRecord, key: string, path: string): number | undefined {
  return key in record ? expectInteger(record[key], `${path}.${key}`) : undefined;
}

function optionalString(record: JsonRecord, key: string, path: string): string | undefined {
  return key in record ? expectString(record[key], `${path}.${key}`) : undefined;
}

function decodeProject(value: unknown, path: string): ModelProject {
  const record = expectRecord(value, path);
  return {
    id: expectInteger(record.id, `${path}.id`),
    name: expectString(record.name, `${path}.name`),
    description: expectString(record.description, `${path}.description`)
  };
}

function decodeVersion(value: unknown, path: string): ModelVersion {
  const record = expectRecord(value, path);
  return {
    id: expectInteger(record.id, `${path}.id`),
    project_id: expectInteger(record.project_id, `${path}.project_id`),
    version: expectString(record.version, `${path}.version`),
    source_kind: expectString(record.source_kind, `${path}.source_kind`),
    source_path: expectString(record.source_path, `${path}.source_path`),
    classes: expectStringArray(record.classes, `${path}.classes`),
    input_shape: expectString(record.input_shape, `${path}.input_shape`)
  };
}

function decodeArtifact(value: unknown, path: string): ModelArtifact {
  const record = expectRecord(value, path);
  return {
    id: expectInteger(record.id, `${path}.id`),
    version_id: expectInteger(record.version_id, `${path}.version_id`),
    kind: expectString(record.kind, `${path}.kind`),
    path: expectString(record.path, `${path}.path`),
    checksum: expectString(record.checksum, `${path}.checksum`),
    status: expectString(record.status, `${path}.status`),
    size_bytes: record.size_bytes === null
      ? null
      : expectUnsignedInteger(record.size_bytes, `${path}.size_bytes`)
  };
}

function decodeDeploymentAt(value: unknown, path: string): Deployment {
  const record = expectRecord(value, path);
  return {
    id: expectInteger(record.id, `${path}.id`),
    project_id: expectInteger(record.project_id, `${path}.project_id`),
    artifact_id: expectInteger(record.artifact_id, `${path}.artifact_id`),
    previous_artifact_id: record.previous_artifact_id === null
      ? null
      : expectInteger(record.previous_artifact_id, `${path}.previous_artifact_id`),
    updated_seq: expectUnsignedInteger(record.updated_seq, `${path}.updated_seq`)
  };
}

function decodeArtifactMetadataAt(value: unknown, path: string): ModelArtifactMetadata {
  const record = expectRecord(value, path);
  return {
    artifact_id: expectInteger(record.artifact_id, `${path}.artifact_id`),
    recommendation: expectRecommendation(record.recommendation, `${path}.recommendation`),
    tags: expectStringArray(record.tags, `${path}.tags`)
  };
}

function decodeParserContract(value: unknown, path: string): ParserContract {
  const record = expectRecord(value, path);
  return {
    requested_preset: expectLiteral(record.requested_preset, `${path}.requested_preset`, [
      "auto",
      "yolov5",
      "yolov8",
      "yolo11",
      "novasight_generic"
    ] as const),
    output_family: expectLiteral(record.output_family, `${path}.output_family`, [
      "yolov5",
      "yolov8_yolo11"
    ] as const),
    has_objectness: expectBoolean(record.has_objectness, `${path}.has_objectness`),
    parser_library: expectLiteral(record.parser_library, `${path}.parser_library`, [
      "novasight_builtin"
    ] as const),
    parser_function: expectLiteral(record.parser_function, `${path}.parser_function`, [
      "NvDsInferParseNovaSight"
    ] as const),
    nms_owner: expectLiteral(record.nms_owner, `${path}.nms_owner`, ["deepstream"] as const)
  };
}

function decodeReportSection(value: unknown, path: string): OperationReportSection {
  const record = expectRecord(value, path);
  return {
    section: expectString(record.section, `${path}.section`),
    impact: expectString(record.impact, `${path}.impact`),
    status: expectString(record.status, `${path}.status`),
    message: expectString(record.message, `${path}.message`)
  };
}

function decodeCatalogModel(value: unknown, path: string): ModelCatalogModel {
  const record = expectRecord(value, path);
  if (record.type !== "model") throw new ModelContractError(`${path}.type`, "model");
  return {
    type: "model",
    name: expectString(record.name, `${path}.name`),
    relative_path: expectString(record.relative_path, `${path}.relative_path`),
    kind: expectString(record.kind, `${path}.kind`),
    size_bytes: expectUnsignedInteger(record.size_bytes, `${path}.size_bytes`),
    scan_status: expectString(record.scan_status, `${path}.scan_status`),
    scan_reason: expectString(record.scan_reason, `${path}.scan_reason`),
    project_id: optionalInteger(record, "project_id", path),
    project_name: optionalString(record, "project_name", path),
    version_id: optionalInteger(record, "version_id", path),
    version_name: optionalString(record, "version_name", path),
    artifact_id: optionalInteger(record, "artifact_id", path),
    artifact_status: optionalString(record, "artifact_status", path),
    recommendation: expectRecommendation(record.recommendation, `${path}.recommendation`),
    tags: expectStringArray(record.tags, `${path}.tags`)
  };
}

function decodeCatalogDirectory(value: unknown, path: string): ModelCatalogDirectory {
  const record = expectRecord(value, path);
  if (record.type !== "directory") {
    throw new ModelContractError(`${path}.type`, "directory");
  }
  if (!Array.isArray(record.children)) {
    throw new ModelContractError(`${path}.children`, "array");
  }
  return {
    type: "directory",
    name: expectString(record.name, `${path}.name`),
    relative_path: expectString(record.relative_path, `${path}.relative_path`),
    children: record.children.map((child, index) => {
      const childPath = `${path}.children[${index}]`;
      const childRecord = expectRecord(child, childPath);
      if (childRecord.type === "directory") return decodeCatalogDirectory(child, childPath);
      if (childRecord.type === "model") return decodeCatalogModel(child, childPath);
      throw new ModelContractError(`${childPath}.type`, "directory | model");
    })
  };
}

function expectArray<T>(
  value: unknown,
  path: string,
  decode: (item: unknown, itemPath: string) => T
): T[] {
  if (!Array.isArray(value)) throw new ModelContractError(path, "array");
  return value.map((item, index) => decode(item, `${path}[${index}]`));
}

export function decodeModelProjects(value: unknown): ModelProject[] {
  return expectArray(value, "model_projects", decodeProject);
}

export function decodeModelVersions(value: unknown): ModelVersion[] {
  return expectArray(value, "model_versions", decodeVersion);
}

export function decodeModelArtifacts(value: unknown): ModelArtifact[] {
  return expectArray(value, "model_artifacts", decodeArtifact);
}

export function decodeDeployment(value: unknown): Deployment {
  return decodeDeploymentAt(value, "deployment");
}

export function decodeActiveModel(value: unknown): ActiveModel {
  const record = expectRecord(value, "active_model");
  return {
    deployment: decodeDeploymentAt(record.deployment, "active_model.deployment"),
    project: decodeProject(record.project, "active_model.project"),
    version: decodeVersion(record.version, "active_model.version"),
    artifact: decodeArtifact(record.artifact, "active_model.artifact"),
    artifact_path: expectString(record.artifact_path, "active_model.artifact_path")
  };
}

export function decodeModelArtifactMetadata(value: unknown): ModelArtifactMetadata {
  return decodeArtifactMetadataAt(value, "model_artifact_metadata");
}

export function decodeModelCatalog(value: unknown): ModelCatalogResponse {
  const record = expectRecord(value, "model_catalog");
  return {
    root: decodeCatalogDirectory(record.root, "model_catalog.root"),
    directory_count: expectUnsignedInteger(
      record.directory_count,
      "model_catalog.directory_count"
    ),
    model_count: expectUnsignedInteger(record.model_count, "model_catalog.model_count"),
    discovered_files: expectUnsignedInteger(
      record.discovered_files,
      "model_catalog.discovered_files"
    ),
    updated_files: expectUnsignedInteger(record.updated_files, "model_catalog.updated_files"),
    cache_hits: expectUnsignedInteger(record.cache_hits, "model_catalog.cache_hits"),
    force: expectBoolean(record.force, "model_catalog.force")
  };
}

export function decodeModelCatalogRegisterResponse(
  value: unknown
): ModelCatalogRegisterResponse {
  const record = expectRecord(value, "model_catalog_registration");
  return {
    project: decodeProject(record.project, "model_catalog_registration.project"),
    version: decodeVersion(record.version, "model_catalog_registration.version"),
    artifact: decodeArtifact(record.artifact, "model_catalog_registration.artifact"),
    engine_path: expectString(record.engine_path, "model_catalog_registration.engine_path"),
    created: expectBoolean(record.created, "model_catalog_registration.created")
  };
}

export function decodeConversionJobs(value: unknown): ConversionJob[] {
  return expectArray(value, "conversion_jobs", (item, path) => {
    const record = expectRecord(item, path);
    return {
      id: expectInteger(record.id, `${path}.id`),
      version_id: expectInteger(record.version_id, `${path}.version_id`),
      target_kind: expectString(record.target_kind, `${path}.target_kind`),
      command: expectStringArray(record.command, `${path}.command`),
      status: expectString(record.status, `${path}.status`),
      log: expectString(record.log, `${path}.log`)
    };
  });
}

export function decodeModelPublishResponse(value: unknown): ModelPublishResponse {
  const path = "model_publish";
  const record = expectRecord(value, path);
  const inference = expectRecord(record.inference, `${path}.inference`);
  const preparation = expectRecord(record.preparation, `${path}.preparation`);
  const report = expectRecord(record.report, `${path}.report`);
  const parserContract = "parser_contract" in record
    ? decodeParserContract(record.parser_contract, `${path}.parser_contract`)
    : undefined;
  return {
    deployment: decodeDeploymentAt(record.deployment, `${path}.deployment`),
    inference: {
      selected: expectString(inference.selected, `${path}.inference.selected`),
      available: expectBoolean(inference.available, `${path}.inference.available`),
      loaded: expectBoolean(inference.loaded, `${path}.inference.loaded`),
      configured: expectBoolean(inference.configured, `${path}.inference.configured`),
      reason: expectString(inference.reason, `${path}.inference.reason`)
    },
    ...(parserContract ? { parser_contract: parserContract } : {}),
    preparation: {
      manifest_action: expectLiteral(
        preparation.manifest_action,
        `${path}.preparation.manifest_action`,
        ["generated", "reused", "unchanged"] as const
      ),
      reason: expectString(preparation.reason, `${path}.preparation.reason`),
      input_shape: expectString(preparation.input_shape, `${path}.preparation.input_shape`),
      classes: expectStringArray(preparation.classes, `${path}.preparation.classes`)
    },
    report: {
      action: expectString(report.action, `${path}.report.action`),
      applied: expectBoolean(report.applied, `${path}.report.applied`),
      rolled_back: expectBoolean(report.rolled_back, `${path}.report.rolled_back`),
      message: expectString(report.message, `${path}.report.message`),
      runtime_error: expectString(report.runtime_error, `${path}.report.runtime_error`),
      artifact_id: expectInteger(report.artifact_id, `${path}.report.artifact_id`),
      previous_artifact_id: report.previous_artifact_id === null
        ? null
        : expectInteger(report.previous_artifact_id, `${path}.report.previous_artifact_id`),
      artifact_path: expectString(report.artifact_path, `${path}.report.artifact_path`),
      backend: expectString(report.backend, `${path}.report.backend`),
      input_shape: expectString(report.input_shape, `${path}.report.input_shape`),
      classes: expectUnsignedInteger(report.classes, `${path}.report.classes`),
      sections: expectArray(report.sections, `${path}.report.sections`, decodeReportSection)
    }
  };
}
