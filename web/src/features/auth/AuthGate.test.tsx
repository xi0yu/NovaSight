import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AuthGate, useWorkspaceAuthorization } from "./AuthGate";

const api = vi.hoisted(() => ({
  getAuthSession: vi.fn(), establishAuthSession: vi.fn(), getHealth: vi.fn(), logoutAuthSession: vi.fn(),
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
  it("preserves workspace draft and pauses authorization while reauthenticating", async () => {
    function Draft() {
      const [value, setValue] = useState("original");
      const authorized = useWorkspaceAuthorization();
      return <input aria-label="草稿" data-authorized={authorized} value={value} onChange={(event) => setValue(event.target.value)} />;
    }
    api.getAuthSession.mockResolvedValue(activeSession);
    api.establishAuthSession.mockResolvedValue(activeSession);
    const view = render(<AuthGate><Draft /></AuthGate>);
    const draft = await screen.findByLabelText("草稿");
    fireEvent.change(draft, { target: { value: "unsaved" } });
    fireEvent(window, new CustomEvent("novasight:auth-required"));
    expect(await screen.findByLabelText("授权码")).toBeVisible();
    expect(draft).toHaveValue("unsaved");
    expect(draft).not.toBeVisible();
    expect(draft).toHaveAttribute("data-authorized", "false");
    await userEvent.type(screen.getByLabelText("授权码"), "renewed");
    await userEvent.click(screen.getByRole("button", { name: "进入控制台" }));
    await waitFor(() => expect(draft).toBeVisible());
    expect(screen.getByLabelText("草稿")).toBe(draft);
    expect(draft).toHaveValue("unsaved");
    expect(draft).toHaveAttribute("data-authorized", "true");
    view.unmount();
  });
  beforeEach(() => {
    vi.clearAllMocks();
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

  it("keeps the workspace open when logout fails instead of claiming the session ended", async () => {
    api.getAuthSession.mockResolvedValue(activeSession);
    api.logoutAuthSession.mockRejectedValue(new Error("device refused logout"));
    render(<AuthGate><div>识别工作台</div></AuthGate>);
    await userEvent.click(await screen.findByRole("button", { name: "退出会话" }));
    await waitFor(() => expect(api.logoutAuthSession).toHaveBeenCalledTimes(1));
    expect(screen.getByText("识别工作台")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "输入授权码" })).not.toBeInTheDocument();
  });

  it("clears an old input error once a new authorization code is entered", async () => {
    render(<AuthGate><div>识别工作台</div></AuthGate>);
    await userEvent.click(await screen.findByRole("button", { name: "进入控制台" }));
    expect(screen.getByRole("alert")).toHaveTextContent("请输入授权码");
    await userEvent.type(screen.getByLabelText("授权码"), "replacement");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
