import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { LicenseStatus } from "../../api";
import { LicenseView } from "./LicenseView";

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
  it("requires confirmation before clearing and returns to the license gate", async () => {
    clearLicenseKeyMock.mockResolvedValueOnce(clearedLicense);
    const user = userEvent.setup();
    render(<LicenseTransitionHarness />);

    expect(screen.getByRole("heading", { name: "当前授权" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "退出当前授权" }));
    expect(clearLicenseKeyMock).not.toHaveBeenCalled();
    expect(screen.getByText(/再次点击将先紧急停止当前设备/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "确认：先紧急停止设备，再退出授权" }));

    expect(await screen.findByText("等待重新授权")).toBeInTheDocument();
    expect(clearLicenseKeyMock).toHaveBeenCalledTimes(1);
  });
});
