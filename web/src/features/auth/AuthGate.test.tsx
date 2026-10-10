import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AuthGate, useWorkspaceAuthorization } from "./AuthGate";

const api = vi.hoisted(() => ({
  getAuthSession: vi.fn(), recoverAuthSession: vi.fn(), establishAuthSession: vi.fn(), getHealth: vi.fn(), logoutAuthSession: vi.fn(),
}));
vi.mock("../../api", async (original) => ({
  ...await original<typeof import("../../api")>(), ...api,
}));
const activeSession = {
  authenticated: true, principal: "local-operator", role: "operator", permissions: [],
  csrf_token: "session-token", expires_at: Math.floor(Date.now() / 1000) + 3600,
  session_lifetime_seconds: 3600,
};

describe("single authorization entry", () => {
  it("keeps the original page and draft while checking a short-lived session", async () => {
    function Draft() {
      const [value, setValue] = useState("original");
      const authorized = useWorkspaceAuthorization();
      return <input aria-label="草稿" data-authorized={authorized} value={value} onChange={(event) => setValue(event.target.value)} />;
    }
    api.getAuthSession.mockResolvedValue(activeSession);
    const view = render(<AuthGate><Draft /></AuthGate>);
    const draft = await screen.findByLabelText("草稿");
    fireEvent.change(draft, { target: { value: "unsaved" } });
    fireEvent(window, new CustomEvent("novasight:auth-required"));
    await waitFor(() => expect(api.recoverAuthSession).toHaveBeenCalledTimes(2));
    expect(draft).toBeVisible();
    expect(draft).toHaveValue("unsaved");
    expect(screen.queryByLabelText("授权码")).not.toBeInTheDocument();
    view.unmount();
  });
  beforeEach(() => {
    vi.clearAllMocks();
    api.recoverAuthSession.mockImplementation(() => api.getAuthSession());
    api.getHealth.mockResolvedValue({ ok: true });
    api.getAuthSession.mockResolvedValue({ ...activeSession, authenticated: false });
  });

  it("submits one license and enters the workspace without a second prompt", async () => {
    api.establishAuthSession.mockResolvedValue(activeSession);
    render(<AuthGate><div>识别工作台</div></AuthGate>);
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("授权码"), "valid-license");
    await user.click(screen.getByRole("button", { name: "进入控制台" }));
    expect(await screen.findByText("识别工作台")).toBeInTheDocument();
    expect(api.establishAuthSession).toHaveBeenCalledTimes(1);
    expect(api.establishAuthSession).toHaveBeenCalledWith("valid-license");
    expect(screen.queryByLabelText("授权码")).not.toBeInTheDocument();
  });

  it("reuses an existing browser session without submitting another code", async () => {
    api.getAuthSession.mockResolvedValue(activeSession);
    render(<AuthGate><div>识别工作台</div></AuthGate>);
    await waitFor(() => expect(screen.getByText("识别工作台")).toBeInTheDocument());
    expect(api.establishAuthSession).not.toHaveBeenCalled();
  });

  it("stops local automatic login on explicit logout without claiming server revocation", async () => {
    api.getAuthSession.mockResolvedValue(activeSession);
    api.logoutAuthSession.mockRejectedValue(new Error("device refused logout"));
    render(<AuthGate><div>识别工作台</div></AuthGate>);
    await userEvent.click(await screen.findByRole("button", { name: "退出登录" }));
    await waitFor(() => expect(api.logoutAuthSession).toHaveBeenCalledTimes(1));
    expect(await screen.findByRole("heading", { name: "输入授权码" })).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("服务端撤销尚未确认");
  });

  it("clears an old input error once a new authorization code is entered", async () => {
    render(<AuthGate><div>识别工作台</div></AuthGate>);
    await userEvent.click(await screen.findByRole("button", { name: "进入控制台" }));
    expect(screen.getByRole("alert")).toHaveTextContent("请输入授权码");
    await userEvent.type(screen.getByLabelText("授权码"), "replacement");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
  it.each(["offline", "restart", "wake"])("preserves the draft during %s and never constructs an authorization terminal", async (trigger) => {
    api.getAuthSession.mockResolvedValue(activeSession);
    render(<AuthGate><input aria-label="草稿" defaultValue="unsaved" /></AuthGate>);
    const draft = await screen.findByLabelText("草稿");
    if (trigger === "offline") fireEvent(window, new Event("offline"));
    else {
      api.getAuthSession.mockRejectedValueOnce(new TypeError("service unavailable"));
      fireEvent(window, new Event(trigger === "wake" ? "focus" : "novasight:auth-required"));
    }
    await screen.findByText("正在恢复连接，页面内容已保留");
    expect(draft).toBeVisible();
    expect(screen.queryByLabelText("授权码")).not.toBeInTheDocument();
    fireEvent(window, new Event("online"));
    await waitFor(() => expect(document.querySelector(".authenticated-workspace")).toHaveAttribute("data-auth-state", "authorized"));
    expect(draft).toHaveValue("unsaved");
  });

  it("shows authorization only after a server-confirmed expiration", async () => {
    api.getAuthSession.mockResolvedValue(activeSession);
    render(<AuthGate><div>原页面</div></AuthGate>);
    await screen.findByText("原页面");
    fireEvent(window, new CustomEvent("novasight:auth-session", { detail: {
      ...activeSession, authenticated: false, auth_state: "expired", principal: null, role: null, csrf_token: null, expires_at: null,
    } }));
    expect(await screen.findByLabelText("授权码")).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("授权已到期");
  });

});
