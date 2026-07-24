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

export function describeLicenseConnectionIssue(error: unknown): LicenseConnectionIssue {
  if (error instanceof ApiError) {
    if (error.status === 408 || error.status === 504) {
      return {
        kind: "timeout",
        title: "NovaSight 后端响应超时",
        description: "本机服务没有在预期时间内返回授权状态，这不代表卡密无效。",
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
        title: "NovaSight 后端暂时异常",
        description: "后端已经响应，但目前无法提供授权状态，这不代表卡密无效。",
        recovery: "请查看 novasightd 运行终端中的最新日志，恢复服务后重新连接。"
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
  action: "activate" | "clear"
): LicenseActionFailure {
  if (error instanceof ApiError) {
    if (error.status === 404 || error.status === 405 || error.status === 408) {
      const issue = describeLicenseConnectionIssue(error);
      return { title: issue.title, message: `${issue.description} ${issue.recovery}` };
    }
    if (error.status >= 400 && error.status < 500) {
      return action === "activate"
        ? {
            title: "授权码未通过校验",
            message: "请检查授权码是否完整、是否过期，以及是否适用于当前设备。"
          }
        : {
            title: "授权清除未完成",
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
