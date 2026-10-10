import { useState } from "react";
import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { LicenseStatus } from "../../api";
import { LicensePanel } from "./LicensePanel";

const { clearLicenseKeyMock, getLicenseStatusMock } = vi.hoisted(() => ({
  clearLicenseKeyMock: vi.fn(),
  getLicenseStatusMock: vi.fn(),
}));

vi.mock("../../api", async (importOriginal) => ({
  ...await importOriginal<typeof import("../../api")>(),
  clearLicenseKey: clearLicenseKeyMock,
  getLicenseStatus: getLicenseStatusMock,
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
    ? <LicensePanel license={license} onLicenseChange={setLicense} />
    : <div role="status">等待重新授权</div>;
}

describe("LicensePanel", () => {
  it("uses a local deadline only to request server confirmation", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-09T00:00:00Z"));
    const changed = vi.fn();
    getLicenseStatusMock.mockResolvedValue(activeLicense);
    const license = { ...activeLicense, device_model: "Mac16,11", expires_at: Date.now() / 1000 + 2 };
    const view = render(<LicensePanel license={license} onLicenseChange={changed} />);
    try {
      expect(screen.getByText("Mac16,11")).toBeVisible();
      expect(screen.getByText("倒计时 00:00:02")).toBeVisible();
      act(() => vi.advanceTimersByTime(2000));
      expect(screen.getByText("正在核验到期状态")).toBeVisible();
      await act(async () => {});
      expect(changed).toHaveBeenCalledWith(activeLicense);
      act(() => vi.advanceTimersByTime(2000));
      expect(changed).toHaveBeenCalledTimes(1);
    } finally {
      view.unmount();
      vi.useRealTimers();
    }
  });
  it.each([
    { credential_format: "ephemeral_code", tier: "temporary", category: "测试" },
    { credential_format: "jwt_rs256", tier: "pro", category: "正式" },
    { credential_format: "legacy_ns1", tier: "premium", category: "正式" },
    { credential_format: "legacy_document", tier: "pro", category: "待核实" },
  ])("shows license metadata rather than a tier title for $credential_format", ({ credential_format, tier, category }) => {
    render(<LicensePanel license={{ ...activeLicense, credential_format, tier, device_model: "Mac16,11" }} onLicenseChange={vi.fn()} />);
    const metadata = within(screen.getByLabelText("设备授权状态"));
    expect(metadata.getByText("设备模型 ID")).toBeVisible();
    expect(metadata.getByText("Mac16,11")).toBeVisible();
    expect(metadata.getByText("到期时间")).toBeVisible();
    expect(metadata.getByText("授权类别")).toBeVisible();
    expect(metadata.getByText(category)).toBeVisible();
    expect(screen.queryByText("临时测试授权")).not.toBeInTheDocument();
    expect(screen.queryByText("专业版")).not.toBeInTheDocument();
  });
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

    expect(screen.getByLabelText("设备授权状态")).toBeInTheDocument();
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
