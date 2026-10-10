import { useState } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it } from "vitest";
import { FireStabilizationPage } from "./FireStabilizationPage";

it("explains a stopped device honestly while the illustration follows the chosen strength and switch", async () => {
  function Workspace() {
    const [enabled, setEnabled] = useState(true);
    const [strength, setStrength] = useState(0.5);
    return <FireStabilizationPage enabled={enabled} strength={strength} runtimeStatus="设备已停止" onEnabled={setEnabled} onStrength={setStrength} />;
  }
  const view = render(<Workspace />);
  expect(screen.getByText("设备已停止")).toBeVisible();
  expect(screen.getByText(/机制示意，非设备实测曲线/)).toBeVisible();
  const path = () => view.container.querySelector(".fire-preview-adjusted")?.getAttribute("d");
  const raw = () => view.container.querySelector(".fire-preview-raw")?.getAttribute("d");
  const before = path();
  const input = screen.getByRole("textbox", { name: "抑制强度 数值" });
  fireEvent.change(input, { target: { value: "0" } });
  fireEvent.blur(input);
  await waitFor(() => expect(path()).toBe(raw()));
  fireEvent.change(input, { target: { value: "1" } });
  fireEvent.blur(input);
  await waitFor(() => expect(path()).not.toBe(before));
  expect(path()).not.toBe(raw());
  await userEvent.click(screen.getByRole("button", { name: /开火时稳定/ }));
  await waitFor(() => expect(path()).toBe(raw()));
  expect(screen.getByText("设备已停止")).toBeVisible();
});
