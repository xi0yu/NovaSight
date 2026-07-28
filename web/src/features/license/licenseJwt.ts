import { InvalidTokenError, jwtDecode, type JwtHeader, type JwtPayload } from "jwt-decode";

const LICENSE_ISSUER = "novasight-license";
const LICENSE_AUDIENCE = "novasightd";
const MAX_CREDENTIAL_LENGTH = 64 * 1024;
const SUPPORTED_FEATURES = new Set([
  "capture",
  "runtime",
  "models",
  "tensorrt",
  "hardware_control",
  "config_read",
  "config_write"
]);

type LicenseClaims = JwtPayload & {
  aud?: string | string[];
  jti?: string;
  tier?: string;
  features?: unknown;
};

export type LicenseCredentialPreview = {
  kind: "jwt" | "legacy" | "invalid";
  acceptedShape: boolean;
  label: string;
  licenseId: string;
  tokenId: string;
  keyId: string;
  tier: string;
  features: string[];
  issuedAt: number | null;
  notBefore: number | null;
  expiresAt: number | null;
  problems: string[];
};

/**
 * Decode only for human-readable preview. This function never verifies a
 * signature and its result must never be used to authorize UI or API access.
 */
export function inspectLicenseCredential(rawCredential: string): LicenseCredentialPreview | null {
  const credential = rawCredential.trim();
  if (!credential) {
    return null;
  }
  if (credential.length > MAX_CREDENTIAL_LENGTH) {
    return invalidPreview("授权凭证超过 64 KiB 限制");
  }
  if (credential.startsWith("NS1.")) {
    const acceptedShape = credential.split(".").length === 3;
    return {
      kind: "legacy",
      acceptedShape,
      label: "旧版 NS1（兼容）",
      licenseId: "",
      tokenId: "",
      keyId: "",
      tier: "",
      features: [],
      issuedAt: null,
      notBefore: null,
      expiresAt: null,
      problems: acceptedShape ? [] : ["旧版 NS1 凭证必须包含三个片段"]
    };
  }
  if (credential.split(".").length !== 3) {
    return invalidPreview("JWT 必须包含 header、payload、signature 三个片段");
  }

  try {
    const header = jwtDecode<JwtHeader>(credential, { header: true });
    const claims = jwtDecode<LicenseClaims>(credential);
    const problems: string[] = [];
    const features = Array.isArray(claims.features)
      ? claims.features.filter((feature): feature is string => typeof feature === "string")
      : [];
    const audience = Array.isArray(claims.aud) ? claims.aud : [claims.aud];

    if (header.alg !== "RS256") {
      problems.push(`签名算法必须是 RS256，当前为 ${header.alg || "未声明"}`);
    }
    if (header.typ && header.typ.toUpperCase() !== "JWT") {
      problems.push("typ 必须是 JWT");
    }
    if (claims.iss !== LICENSE_ISSUER) {
      problems.push("签发方不是 NovaSight 授权服务");
    }
    if (!audience.includes(LICENSE_AUDIENCE)) {
      problems.push("授权凭证不适用于 novasightd");
    }
    if (!claims.sub?.trim() || !claims.jti?.trim() || !claims.tier?.trim()) {
      problems.push("缺少 sub、jti 或 tier 授权字段");
    }
    if (!Number.isFinite(claims.iat) || !Number.isFinite(claims.exp)) {
      problems.push("缺少有效的 iat 或 exp 时间字段");
    } else if ((claims.exp ?? 0) <= (claims.iat ?? 0)) {
      problems.push("到期时间必须晚于签发时间");
    }
    if (typeof claims.exp === "number" && claims.exp * 1000 <= Date.now()) {
      problems.push("该授权凭证已经到期");
    }
    if (claims.nbf && claims.nbf * 1000 > Date.now()) {
      problems.push("该授权凭证尚未到生效时间");
    }
    if (!Array.isArray(claims.features) || features.length !== claims.features.length) {
      problems.push("features 必须是字符串数组");
    } else if (new Set(features).size !== features.length) {
      problems.push("features 不能包含重复权限");
    } else if (features.some((feature) => !SUPPORTED_FEATURES.has(feature))) {
      problems.push("features 包含当前版本不支持的权限");
    }

    return {
      kind: "jwt",
      acceptedShape: problems.length === 0,
      label: "JWT · RS256",
      licenseId: claims.sub ?? "",
      tokenId: claims.jti ?? "",
      keyId: header.kid ?? "",
      tier: claims.tier ?? "",
      features,
      issuedAt: finiteNumber(claims.iat),
      notBefore: finiteNumber(claims.nbf),
      expiresAt: finiteNumber(claims.exp),
      problems
    };
  } catch (error) {
    return invalidPreview(
      error instanceof InvalidTokenError ? "JWT 编码无法解析" : "授权凭证无法解析"
    );
  }
}

function finiteNumber(value: number | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function invalidPreview(message: string): LicenseCredentialPreview {
  return {
    kind: "invalid",
    acceptedShape: false,
    label: "无法识别",
    licenseId: "",
    tokenId: "",
    keyId: "",
    tier: "",
    features: [],
    issuedAt: null,
    notBefore: null,
    expiresAt: null,
    problems: [message]
  };
}
