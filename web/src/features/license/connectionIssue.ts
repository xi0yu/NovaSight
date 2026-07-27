import { ApiError } from "../../api";

export type LicenseConnectionIssue = {
  kind: "unreachable" | "timeout" | "incompatible" | "service-error";
  title: string;
  description: string;
  recovery: string;
};

export type LicenseActionFailure = {
  title: string;
  message: string;
};

function apiErrorCode(error: ApiError): string {
  const detail = typeof error.detail === "object" && error.detail !== null
    ? error.detail as Record<string, unknown>
    : {};
  return typeof detail.code === "string" ? detail.code : "";
}

export function describeLicenseConnectionIssue(error: unknown): LicenseConnectionIssue {
  if (error instanceof ApiError) {
    const code = apiErrorCode(error);
    if (code === "LICENSE_STORAGE_FAILED" || code === "LICENSE_TASK_FAILED") {
      return {
        kind: "service-error",
        title: "授权存储暂时不可用",
        description: "novasightd 在线，但授权文件或锁文件目前无法读取；这不是临时授权被拒绝。",
        recovery: "请检查配置中的 paths.license 及其父目录权限，然后重新读取授权状态。"
      };
    }
    if (code === "LICENSE_PUBLIC_KEY_MISSING" || code === "LICENSE_PUBLIC_KEY_INVALID") {
      return {
        kind: "service-error",
        title: "授权验证器尚未配置完成",
        description: "novasightd 在线，但生产授权公钥缺失或格式无效；这不是当前临时授权被拒绝。",
        recovery: "请修复 NOVASIGHT_LICENSE_PUBLIC_KEY 配置并重启 novasightd。"
      };
    }
    if (code === "LICENSE_SERVICE_UNAVAILABLE") {
      return {
        kind: "service-error",
        title: "授权服务未挂载",
        description: "当前 novasightd 没有启用授权存储服务，因此无法读取授权状态。",
        recovery: "请确认使用完整的 novasightd 服务入口启动，而不是仅启动局部 API。"
      };
    }
    if (error.status === 408 || error.status === 504) {
      return {
        kind: "timeout",
        title: "NovaSight 后端响应超时",
        description: "本机服务没有在预期时间内返回授权状态，这不代表授权申请失败。",
        recovery: "请确认 Jetson 负载正常且 novasightd 仍在运行，然后重新连接。"
      };
    }
    if (error.status === 404 || error.status === 405) {
      return {
        kind: "incompatible",
        title: "NovaSight 后端接口不可用",
        description: "前端已连接到本机服务，但当前后端没有提供所需的授权接口。",
        recovery: "请确认前端与 novasightd 来自同一版本，然后重新启动后端。"
      };
    }
    if (error.status >= 500) {
      return {
        kind: "service-error",
        title: "授权状态暂时无法读取",
        description: "novasightd 已响应，但授权子系统返回了服务错误；采集或推理后端不一定异常。",
        recovery: "请查看异常信息中的授权错误代码，并检查授权文件、公钥及目录权限。"
      };
    }
  }

  return {
    kind: "unreachable",
    title: "尚未连接到 NovaSight 后端",
    description: "浏览器无法读取本机服务，因此现在还不能判断授权状态。",
    recovery: "请先启动 novasightd，并确认本机后端地址可以访问，然后重新连接。"
  };
}

export function describeLicenseActionFailure(
  error: unknown,
  action: "temporary" | "activate" | "clear"
): LicenseActionFailure {
  if (error instanceof ApiError) {
    if (error.status === 404 || error.status === 405 || error.status === 408) {
      const issue = describeLicenseConnectionIssue(error);
      return { title: issue.title, message: `${issue.description} ${issue.recovery}` };
    }
    if (error.status >= 400 && error.status < 500) {
      if (action === "temporary") {
        return {
          title: "临时授权暂不可用",
          message: "当前后端没有批准临时授权，请确认正在运行 Debug 测试版本后重试。"
        };
      }
      if (action === "activate") {
        return {
          title: "正式授权未通过校验",
          message: "请检查授权凭证是否完整、是否过期，以及是否适用于当前版本。"
        };
      }
      return {
        title: "授权退出未完成",
        message: "当前授权状态无法变更，请刷新状态后再试。"
      };
    }
  }

  const issue = describeLicenseConnectionIssue(error);
  return {
    title: issue.title,
    message: `${issue.description} ${issue.recovery}`
  };
}
