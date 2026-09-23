import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { LicenseStatus } from "../../api";
import { LicenseView } from "./LicenseView";
import { LicensePanel } from "./LicensePanel";

const { clearLicenseKeyMock } = vi.hoisted(() => ({
  clearLicenseKeyMock: vi.fn(),
}));

vi.mock("../../api", async (importOriginal) => ({
  ...await importOriginal<typeof import("../../api")>(),
  clearLicenseKey: clearLicenseKeyMock,
}));

const activeLicense: LicenseStatus = {
  configured: true,
  valid: true,
  temporary_access_supported: true,
  fingerprint: "credential-fingerprint",
  tier: "pro",
  features: ["capture", "runtime", "models", "config_read", "config_write"],
  license_id: "license-a",
  credential_format: "jwt_rs256",
  token_id: "token-a",
  key_id: "key-a",
  created_at: 1,
  not_before: 1,
  activated_at: 1,
  expires_at: null,
  duration_value: null,
  duration_unit: "",
  updated_at: 1,
  message: "",
};

const clearedLicense: LicenseStatus = {
  ...activeLicense,
  configured: false,
  valid: false,
  fingerprint: "",
  tier: "",
  features: [],
  license_id: "",
  credential_format: "",
  token_id: "",
  key_id: "",
  created_at: null,
  not_before: null,
  activated_at: null,
  updated_at: null,
};

function LicenseTransitionHarness() {
  const [license, setLicense] = useState(activeLicense);
  return license.valid
    ? <LicenseView license={license} onLicenseChange={setLicense} />
    : <div role="status">等待重新授权</div>;
}

describe("LicenseView", () => {
  it("removes a stale authorization error while the operator corrects the code", async () => {
    render(<LicensePanel license={activeLicense} onLicenseChange={vi.fn()} />);
    await userEvent.click(screen.getByText("更换授权码"));
    await userEvent.click(screen.getByRole("button", { name: "验证并更换" }));
    expect(screen.getByText("授权码不能为空。")).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("授权码"), "replacement");
    expect(screen.queryByText("授权码不能为空。")).not.toBeInTheDocument();
  });
  it("requires confirmation before clearing and returns to the license gate", async () => {
    clearLicenseKeyMock.mockResolvedValueOnce(clearedLicense);
    const user = userEvent.setup();
    render(<LicenseTransitionHarness />);

    expect(screen.getByLabelText("当前授权")).toBeInTheDocument();
    expect(screen.getByText("更换授权码")).toBeInTheDocument();
    expect(screen.getByText("查看授权范围与有效期")).toBeInTheDocument();
    expect(screen.getByText("开发者诊断与凭证追踪码")).not.toBeVisible();
    await user.click(screen.getByText("查看授权范围与有效期"));
    expect(screen.getByText("开发者诊断与凭证追踪码")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "退出当前授权" }));
    expect(clearLicenseKeyMock).not.toHaveBeenCalled();
    expect(screen.getByText(/再次点击将先紧急停止当前设备/)).toHaveAttribute("role", "status");

    await user.click(screen.getByRole("button", { name: "确认：先紧急停止设备，再退出授权" }));

    expect(await screen.findByText("等待重新授权")).toBeInTheDocument();
    expect(clearLicenseKeyMock).toHaveBeenCalledTimes(1);
  });
});
