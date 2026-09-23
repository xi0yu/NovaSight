export interface CaptureCapability {
  pixel_format: string;
  width: number;
  height: number;
  fps_list: number[];
}

export interface CaptureCapabilitiesResponse {
  available: boolean;
  device: string;
  capabilities: CaptureCapability[];
  reason: string;
}

export interface CaptureSelectPayload {
  device: string;
  preference?: "auto_high_fps" | "auto_low_latency" | "auto_balanced" | "manual";
  pixel_format?: string;
  width?: number;
  height?: number;
  fps?: number;
}

export class CaptureContractError extends Error {
  constructor(path: string, expected: string) {
    super(`采集能力数据契约错误：${path} 应为 ${expected}`);
    this.name = "CaptureContractError";
  }
}

function expectRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new CaptureContractError(path, "object");
  }
  return value as Record<string, unknown>;
}

function expectString(value: unknown, path: string): string {
  if (typeof value !== "string") throw new CaptureContractError(path, "string");
  return value;
}

function expectBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new CaptureContractError(path, "boolean");
  return value;
}

function expectU32(value: unknown, path: string): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < 0 ||
    value > 0xffff_ffff
  ) {
    throw new CaptureContractError(path, "u32");
  }
  return value;
}

function decodeCapability(value: unknown, path: string): CaptureCapability {
  const record = expectRecord(value, path);
  if (!Array.isArray(record.fps_list)) {
    throw new CaptureContractError(`${path}.fps_list`, "u32[]");
  }
  return {
    pixel_format: expectString(record.pixel_format, `${path}.pixel_format`),
    width: expectU32(record.width, `${path}.width`),
    height: expectU32(record.height, `${path}.height`),
    fps_list: record.fps_list.map((fps, index) => expectU32(fps, `${path}.fps_list[${index}]`))
  };
}

export function decodeCaptureCapabilities(value: unknown): CaptureCapabilitiesResponse {
  const record = expectRecord(value, "capture_capabilities");
  if (!Array.isArray(record.capabilities)) {
    throw new CaptureContractError("capture_capabilities.capabilities", "array");
  }
  return {
    available: expectBoolean(record.available, "capture_capabilities.available"),
    device: expectString(record.device, "capture_capabilities.device"),
    capabilities: record.capabilities.map((capability, index) =>
      decodeCapability(capability, `capture_capabilities.capabilities[${index}]`)
    ),
    reason: expectString(record.reason, "capture_capabilities.reason")
  };
}
