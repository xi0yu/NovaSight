export interface DeviceReceipt {
  attempt: number;
  epoch: number;
  generation: number;
  issued_at: number;
  target_object_id: number;
  delta_x_counts: number;
  delta_y_counts: number;
}

export interface DiagnosticDeviceStatus {
  available: boolean;
  connected: boolean;
  connection_state: string;
  move_count: number;
  last_dx: number;
  last_dy: number;
  managed_by_runtime: boolean;
}

export interface DiagnosticMoveResponse {
  sent: boolean;
  queued: boolean;
  steps_sent: number;
  message: string;
  receipt: DeviceReceipt;
  status: DiagnosticDeviceStatus;
  metadata: {
    api_name: string;
  };
}

export class HardwareContractError extends Error {
  constructor(path: string, expected: string) {
    super(`硬件输出数据契约错误：${path} 应为 ${expected}`);
    this.name = "HardwareContractError";
  }
}

function expectRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new HardwareContractError(path, "object");
  }
  return value as Record<string, unknown>;
}

function expectString(value: unknown, path: string): string {
  if (typeof value !== "string") throw new HardwareContractError(path, "string");
  return value;
}

function expectBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new HardwareContractError(path, "boolean");
  return value;
}

function expectInteger(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new HardwareContractError(path, "safe integer");
  }
  return value;
}

function expectUnsignedInteger(value: unknown, path: string): number {
  const number = expectInteger(value, path);
  if (number < 0) throw new HardwareContractError(path, "unsigned safe integer");
  return number;
}

function expectI32(value: unknown, path: string): number {
  const number = expectInteger(value, path);
  if (number < -0x8000_0000 || number > 0x7fff_ffff) {
    throw new HardwareContractError(path, "i32");
  }
  return number;
}

function decodeReceipt(value: unknown, path: string): DeviceReceipt {
  const record = expectRecord(value, path);
  return {
    attempt: expectUnsignedInteger(record.attempt, `${path}.attempt`),
    epoch: expectUnsignedInteger(record.epoch, `${path}.epoch`),
    generation: expectUnsignedInteger(record.generation, `${path}.generation`),
    issued_at: expectUnsignedInteger(record.issued_at, `${path}.issued_at`),
    target_object_id: expectUnsignedInteger(record.target_object_id, `${path}.target_object_id`),
    delta_x_counts: expectI32(record.delta_x_counts, `${path}.delta_x_counts`),
    delta_y_counts: expectI32(record.delta_y_counts, `${path}.delta_y_counts`)
  };
}

export function decodeDiagnosticMoveResponse(value: unknown): DiagnosticMoveResponse {
  const path = "diagnostic_move";
  const record = expectRecord(value, path);
  const status = expectRecord(record.status, `${path}.status`);
  const metadata = expectRecord(record.metadata, `${path}.metadata`);
  return {
    sent: expectBoolean(record.sent, `${path}.sent`),
    queued: expectBoolean(record.queued, `${path}.queued`),
    steps_sent: expectUnsignedInteger(record.steps_sent, `${path}.steps_sent`),
    message: expectString(record.message, `${path}.message`),
    receipt: decodeReceipt(record.receipt, `${path}.receipt`),
    status: {
      available: expectBoolean(status.available, `${path}.status.available`),
      connected: expectBoolean(status.connected, `${path}.status.connected`),
      connection_state: expectString(status.connection_state, `${path}.status.connection_state`),
      move_count: expectUnsignedInteger(status.move_count, `${path}.status.move_count`),
      last_dx: expectI32(status.last_dx, `${path}.status.last_dx`),
      last_dy: expectI32(status.last_dy, `${path}.status.last_dy`),
      managed_by_runtime: expectBoolean(
        status.managed_by_runtime,
        `${path}.status.managed_by_runtime`
      )
    },
    metadata: {
      api_name: expectString(metadata.api_name, `${path}.metadata.api_name`)
    }
  };
}
