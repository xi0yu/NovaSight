import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
const reportInfo = vi.hoisted(() => vi.fn());
vi.mock("../../lib/toast", async (original) => ({ ...await original<typeof import("../../lib/toast")>(), reportInfo }));
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
    api.establishAuthSession.mockRejectedValue(new Error("正式版不支持空授权码"));
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
    expect(await screen.findByRole("heading", { name: "设备授权" })).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("服务端撤销尚未确认");
  });

  it("clears an old input error once a new authorization code is entered", async () => {
    render(<AuthGate><div>识别工作台</div></AuthGate>);
    await userEvent.click(await screen.findByRole("button", { name: "进入控制台" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("正式版不支持空授权码");
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

  it.each(["", "   "])("submits %j as an explicit internal application", async (code) => {
    api.establishAuthSession.mockResolvedValue(activeSession);
    render(<AuthGate><div>识别工作台</div></AuthGate>);
    fireEvent.change(await screen.findByLabelText("授权码"), { target: { value: code } });
    await userEvent.click(screen.getByRole("button", { name: "进入控制台" }));
    expect(await screen.findByText("识别工作台")).toBeVisible();
    expect(api.establishAuthSession).toHaveBeenCalledWith("");
  });

  it.each([0, 3600, 3601])("uses confirmed machine time for a %i second expiry warning", async (remaining) => {
    const now = Math.floor(Date.now() / 1000);
    api.getAuthSession.mockResolvedValue({ ...activeSession, server_time: now, license_expires_at: now + remaining });
    render(<AuthGate><div>任意工作页面</div></AuthGate>);
    await screen.findByText("任意工作页面");
    expect(reportInfo).toHaveBeenCalledTimes(remaining === 3600 ? 1 : 0);
  });

  it("confirms the warning threshold with the server and deduplicates renewal and page remount", async () => {
    vi.useFakeTimers();
    const now = Math.floor(Date.now() / 1000);
    const deadline = now + 3601;
    api.getAuthSession.mockResolvedValue({ ...activeSession, server_time: now, license_expires_at: deadline });
    const view = render(<AuthGate><div>任意工作页面</div></AuthGate>);
    await act(async () => { await Promise.resolve(); });
    api.getAuthSession.mockRejectedValueOnce(new TypeError("offline"));
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(reportInfo).not.toHaveBeenCalled();
    api.getAuthSession.mockResolvedValue({ ...activeSession, server_time: now + 2, license_expires_at: deadline });
    await act(async () => { fireEvent(window, new Event("online")); });
    expect(reportInfo).toHaveBeenCalledTimes(1);
    await act(async () => { fireEvent(window, new Event("focus")); });
    expect(reportInfo).toHaveBeenCalledTimes(1);
    view.unmount();
    render(<AuthGate><div>重新打开的页面</div></AuthGate>);
    await act(async () => { await Promise.resolve(); });
    expect(reportInfo).toHaveBeenCalledTimes(1);
    await act(async () => { fireEvent(window, new CustomEvent("novasight:auth-session", { detail: { ...activeSession, server_time: now + 100, license_expires_at: deadline + 50 } })); });
    expect(reportInfo).toHaveBeenCalledTimes(2);
  });

  it("defers a background-tab warning until visibility returns and identity is confirmed", async () => {
    const visibility = vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
    try {
      const now = Math.floor(Date.now() / 1000);
      api.getAuthSession.mockImplementation(async () => ({ ...activeSession, server_time: now, license_expires_at: now + 1800 }));
      render(<AuthGate><div>后台页面</div></AuthGate>);
      await screen.findByText("后台页面");
      expect(reportInfo).not.toHaveBeenCalled();
      visibility.mockReturnValue("visible");
      fireEvent(document, new Event("visibilitychange"));
      await waitFor(() => expect(reportInfo).toHaveBeenCalledTimes(1));
    } finally { visibility.mockRestore(); }
  });

});
