export type AuthSession = {
  authenticated: boolean;
  principal: string | null;
  role: "operator" | null;
  permissions: string[];
  csrf_token: string | null;
  expires_at: number | null;
  session_lifetime_seconds: number;
};

export class AuthContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AuthContractError";
  }
}

export function decodeAuthSession(value: unknown): AuthSession {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new AuthContractError("认证会话响应必须是对象");
  }
  const record = value as Record<string, unknown>;
  if (typeof record.authenticated !== "boolean") {
    throw new AuthContractError("认证会话 authenticated 字段无效");
  }
  if (
    typeof record.session_lifetime_seconds !== "number"
    || !Number.isSafeInteger(record.session_lifetime_seconds)
    || record.session_lifetime_seconds <= 0
  ) {
    throw new AuthContractError("认证会话生命周期字段无效");
  }
  if (!Array.isArray(record.permissions) || !record.permissions.every((item) => typeof item === "string")) {
    throw new AuthContractError("认证会话 permissions 字段无效");
  }
  const nullableString = (field: string): string | null => {
    const fieldValue = record[field];
    if (fieldValue === null) return null;
    if (typeof fieldValue !== "string" || fieldValue.trim() === "") {
      throw new AuthContractError(`认证会话 ${field} 字段无效`);
    }
    return fieldValue;
  };
  const principal = nullableString("principal");
  const role = nullableString("role");
  const csrfToken = nullableString("csrf_token");
  const expiresAt = record.expires_at;
  if (expiresAt !== null && (typeof expiresAt !== "number" || !Number.isSafeInteger(expiresAt) || expiresAt <= 0)) {
    throw new AuthContractError("认证会话 expires_at 字段无效");
  }
  if (record.authenticated) {
    if (principal === null || role !== "operator" || csrfToken === null || expiresAt === null) {
      throw new AuthContractError("已认证会话缺少 operator 身份、CSRF 或过期时间");
    }
  } else if (principal !== null || role !== null || csrfToken !== null || expiresAt !== null || record.permissions.length > 0) {
    throw new AuthContractError("未认证会话不得携带身份或权限");
  }
  return {
    authenticated: record.authenticated,
    principal,
    role: role as "operator" | null,
    permissions: [...record.permissions] as string[],
    csrf_token: csrfToken,
    expires_at: expiresAt as number | null,
    session_lifetime_seconds: record.session_lifetime_seconds
  };
}
