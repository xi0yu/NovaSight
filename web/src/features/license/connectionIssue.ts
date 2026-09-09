import { ApiError, getApiErrorCode } from "../../api";

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
    const code = getApiErrorCode(error);
    if (code === "DAEMON_UNAVAILABLE") {
      return {
        kind: "unreachable",
        title: "novasightd 本地控制服务不可达",
        description: "浏览器会话与 Web/API 安全层仍然有效，但 Web/API 无法通过 Unix IPC 读取运行与授权状态。",
        recovery: "请检查 novasightd 是否运行以及 run/novasightd.sock 是否可访问；恢复后点击重新连接。"
      };
    }
    if (code === "LICENSE_STORAGE_FAILED" || code === "LICENSE_TASK_FAILED") {
      return {
        kind: "service-error",
        title: "正式授权存储暂时不可用",
        description: "novasightd 在线，但正式授权文件或锁文件目前无法读取。进程临时授权不会写入该文件。",
        recovery: "请检查 paths.license 及其父目录权限；Debug 临时码仍通过统一授权入口验证。"
      };
    }
    if (code === "LICENSE_PUBLIC_KEY_MISSING" || code === "LICENSE_PUBLIC_KEY_INVALID") {
      return {
        kind: "service-error",
        title: "正式授权验证器尚未配置完成",
        description: "novasightd 在线，但 Release 构建所需的生产授权公钥缺失或格式无效。",
        recovery: "正式环境请修复 NOVASIGHT_LICENSE_PUBLIC_KEY；Debug 构建可使用本次启动生成的临时授权码。"
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
        title: "NovaSight 服务响应超时",
        description: "本机服务没有在预期时间内返回授权状态，这不代表授权申请失败。",
        recovery: "请确认 Jetson 负载正常且 novasightd 仍在运行，然后重新连接。"
      };
    }
    if (error.status === 404 || error.status === 405) {
      return {
        kind: "incompatible",
        title: "NovaSight 服务接口不可用",
        description: "界面已连接到本机服务，但当前服务没有提供所需的授权接口。",
        recovery: "请确认 Web UI 与 novasightd 来自同一版本，然后重新启动 novasightd。"
      };
    }
    if (error.status >= 500 && code === "") {
      return {
        kind: "service-error",
        title: "NovaSight 服务连接仍在建立",
        description: "Web 前端在线，但代理这次没有收到 novasightd 的业务响应；这不是授权凭证错误。",
        recovery: "novasightd 刚启动或重启时会短暂出现；请等待服务恢复后重新提交授权码。"
      };
    }
    if (error.status >= 500) {
      return {
        kind: "service-error",
        title: "授权服务暂时未完成状态检查",
        description: "novasightd 在线，但这次授权状态检查没有成功；这不是授权凭证无效，也不代表采集或推理故障。",
        recovery: "请查看 novasightd 日志中的 LICENSE_ 错误代码后重试。"
      };
    }
  }

  return {
    kind: "unreachable",
    title: "尚未连接到 NovaSight 服务",
    description: "浏览器无法读取本机服务，因此现在还不能判断授权状态。",
    recovery: "请先启动 novasightd，并确认本机服务地址可以访问，然后重新连接。"
  };
}

export function describeLicenseActionFailure(
  error: unknown,
  action: "activate" | "clear"
): LicenseActionFailure {
  if (error instanceof ApiError) {
    if (getApiErrorCode(error) === "LICENSE_ACTIVATION_RATE_LIMITED") {
      return {
        title: "授权验证已暂时限速",
        message: "一分钟内提交次数过多。请等待 60 秒后使用当前有效授权码重试。"
      };
    }
    if (error.status === 404 || error.status === 405 || error.status === 408) {
      const issue = describeLicenseConnectionIssue(error);
      return { title: issue.title, message: `${issue.description} ${issue.recovery}` };
    }
    if (error.status >= 400 && error.status < 500) {
      if (action === "activate") {
        return {
          title: "授权码未通过校验",
          message: "请检查临时授权码是否属于本次启动，或正式许可证是否完整、有效并适用于当前版本。"
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
