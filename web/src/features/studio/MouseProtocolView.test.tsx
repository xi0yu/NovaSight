import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { MouseProtocolView, readMouseProtocols, type MouseProtocol } from "./MouseProtocolView";
import { ToastHost } from "../../components/ToastHost";

const protocol: MouseProtocol = { id: "net-one", model: "net", host: "10.0.0.2", port: 8888, uuid: "A1B2C3D4" };
const base = { protocols: [], selectedId: "empty", available: true, running: false, saving: false, previewOnly: false };

it("selects and enables with cards, and configures only through the toolbar dialog", async () => {
  const onSelect = vi.fn(async () => {}), onProbe = vi.fn();
  const props = { ...base, protocols: [protocol], onSave: vi.fn(), onSelect, onProbe };
  const view = render(<MouseProtocolView {...props} />);
  expect(screen.getByRole("button", { name: "空" })).toHaveAttribute("aria-pressed", "true");
  await userEvent.click(screen.getByRole("button", { name: /kmBox Net ·/ }));
  expect(onSelect).toHaveBeenCalledWith(protocol);
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  view.rerender(<MouseProtocolView {...props} selectedId={protocol.id} />);
  await userEvent.click(screen.getByRole("button", { name: "配置协议" }));
  expect(screen.getByRole("dialog", { name: "配置协议" })).toBeVisible();
  await userEvent.keyboard("{Escape}");
  expect(screen.getByRole("button", { name: "配置协议" })).toHaveFocus();
  await userEvent.click(screen.getByRole("button", { name: "空" }));
  expect(onSelect).toHaveBeenLastCalledWith(null);
  expect(screen.queryByRole("button", { name: "启用协议" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "取消协议" })).not.toBeInTheDocument();
  expect(onProbe).not.toHaveBeenCalled();
});

it.each([{ protocols: [] }, { protocols: [protocol] }])("shows a toast instead of opening configuration for empty, with saved protocols $protocols", async ({ protocols }) => {
  const onSave = vi.fn(), onSelect = vi.fn(), onProbe = vi.fn();
  render(<><MouseProtocolView {...base} protocols={protocols} onSave={onSave} onSelect={onSelect} onProbe={onProbe} /><ToastHost /></>);
  await userEvent.click(screen.getByRole("button", { name: "配置协议" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.getByRole("region", { name: "系统提示" })).toHaveTextContent("空协议无需配置");
  expect(onSave).not.toHaveBeenCalled();
  expect(onSelect).not.toHaveBeenCalled();
  expect(onProbe).not.toHaveBeenCalled();
});

it("validates address, port and UUID, retains failed input, and retries the same protocol id", async () => {
  const onSave = vi.fn().mockRejectedValueOnce(new Error("write failed")).mockResolvedValue(undefined);
  render(<MouseProtocolView {...base} onSave={onSave} onSelect={vi.fn()} onProbe={vi.fn()} />);
  await userEvent.click(screen.getByRole("button", { name: "新增协议" }));
  const dialog = within(screen.getByRole("dialog", { name: "新增协议" }));
  fireEvent.change(dialog.getByRole("textbox", { name: "IP" }), { target: { value: "bad-ip" } });
  await userEvent.click(dialog.getByRole("button", { name: "添加协议" }));
  expect(dialog.getByRole("alert")).toHaveTextContent("IPv4");
  fireEvent.change(dialog.getByRole("textbox", { name: "IP" }), { target: { value: "10.0.0.2" } });
  fireEvent.change(dialog.getByRole("spinbutton", { name: "Port" }), { target: { value: "0" } });
  await userEvent.click(dialog.getByRole("button", { name: "添加协议" }));
  expect(dialog.getByRole("alert")).toHaveTextContent("Port");
  fireEvent.change(dialog.getByRole("spinbutton", { name: "Port" }), { target: { value: "8888" } });
  fireEvent.change(dialog.getByRole("textbox", { name: "UUID" }), { target: { value: "invalid" } });
  await userEvent.click(dialog.getByRole("button", { name: "添加协议" }));
  expect(dialog.getByRole("alert")).toHaveTextContent("UUID");
  expect(onSave).not.toHaveBeenCalled();
  fireEvent.change(dialog.getByRole("textbox", { name: "UUID" }), { target: { value: "A1B2C3D4" } });
  await userEvent.click(dialog.getByRole("button", { name: "添加协议" }));
  expect(await dialog.findByRole("alert")).toHaveTextContent("write failed");
  expect(dialog.getByRole("textbox", { name: "IP" })).toHaveValue("10.0.0.2");
  await userEvent.click(dialog.getByRole("button", { name: "添加协议" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(onSave.mock.calls[0][0]).toEqual(onSave.mock.calls[1][0]);
  expect(onSave.mock.calls[1][0]).toMatchObject({ model: "net_pro8k", host: "10.0.0.2", port: 8888, uuid: "A1B2C3D4" });
});

it("keeps the page to cards and two toolbar actions, configuring in a separate dialog", async () => {
  const view = render(<MouseProtocolView {...base} protocols={[protocol]} selectedId={protocol.id} onSave={vi.fn()} onSelect={vi.fn()} onProbe={vi.fn()} diagnostics={<details><summary>高级诊断</summary></details>} />);
  const workspace = screen.getByRole("region", { name: "鼠标协议管理" });
  expect(within(workspace).getByRole("button", { name: "新增协议" })).toBeVisible();
  expect(within(workspace).getByRole("button", { name: "配置协议" })).toBeVisible();
  expect(within(workspace).getAllByRole("button")).toHaveLength(4);
  expect(screen.queryByText("高级诊断")).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "配置协议" }));
  expect(screen.getByRole("dialog", { name: "配置协议" })).toContainElement(screen.getByText("高级诊断"));
  await userEvent.keyboard("{Escape}");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "配置协议" })).toHaveFocus();
  view.unmount();
});

it("tests draft connection fields without saving or enabling a consumer", async () => {
  const onSave = vi.fn(async () => {});
  const onProbe = vi.fn(async () => ({ elapsed_ms: 12 }));
  const props = { ...base, protocols: [protocol], selectedId: protocol.id, onSave, onSelect: vi.fn(), onProbe };
  const view = render(<MouseProtocolView {...props} />);
  await userEvent.click(screen.getByRole("button", { name: "配置协议" }));
  fireEvent.change(screen.getByRole("textbox", { name: "IP" }), { target: { value: "10.0.0.3" } });
  await userEvent.click(screen.getByRole("button", { name: "测试连通性" }));
  expect(await screen.findByText(/设备握手成功/)).toBeVisible();
  expect(onProbe).toHaveBeenCalledOnce();
  expect(onSave).not.toHaveBeenCalled();
  expect(onProbe).toHaveBeenCalledWith({ ...protocol, host: "10.0.0.3" });
  view.rerender(<MouseProtocolView {...props} selectedId="empty" />);
  expect(screen.getByText(/设备握手成功/)).toBeVisible();
  expect(screen.getByRole("button", { name: "测试连通性" })).toBeEnabled();
  view.rerender(<MouseProtocolView {...props} running />);
  expect(screen.getByRole("textbox", { name: "IP" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "空" })).toBeDisabled();
});

it("keeps failed candidate test input without saving", async () => {
  const onProbe = vi.fn(async () => { throw new Error("handshake failed"); });
  render(<MouseProtocolView {...base} protocols={[protocol]} selectedId={protocol.id} onSave={vi.fn(async () => { throw new Error("write failed"); })} onSelect={vi.fn()} onProbe={onProbe} />);
  await userEvent.click(screen.getByRole("button", { name: "配置协议" }));
  fireEvent.change(screen.getByRole("textbox", { name: "IP" }), { target: { value: "10.0.0.3" } });
  await userEvent.click(screen.getByRole("button", { name: "测试连通性" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("handshake failed");
  expect(screen.getByRole("textbox", { name: "IP" })).toHaveValue("10.0.0.3");
  expect(onProbe).toHaveBeenCalledWith({ ...protocol, host: "10.0.0.3" });
});

it("preserves legacy connections and rejects unknown registry model names", () => {
  expect(readMouseProtocols({ ...protocol, backend: "native_udp", mouse_protocols: [] })).toEqual([{ ...protocol, id: "legacy" }]);
  expect(readMouseProtocols({ backend: "null" })).toEqual([]);
  expect(readMouseProtocols({ mouse_protocols: [{ ...protocol, model: "toString" }] })).toEqual([]);
});

it("uses the displayed defaults for blank fields including whitespace and a cleared port", async () => {
  const onSave = vi.fn(async () => {});
  render(<MouseProtocolView {...base} onSave={onSave} onSelect={vi.fn()} onProbe={vi.fn()} />);
  await userEvent.click(screen.getByRole("button", { name: "新增协议" }));
  expect(screen.getByRole("textbox", { name: "IP" })).toHaveAttribute("placeholder", "192.168.2.188");
  expect(screen.getByRole("spinbutton", { name: "Port" })).toHaveAttribute("placeholder", "8888");
  expect(screen.getByRole("textbox", { name: "UUID" })).toHaveAttribute("placeholder", "A1B2C3D4");
  fireEvent.change(screen.getByRole("textbox", { name: "IP" }), { target: { value: "  " } });
  fireEvent.change(screen.getByRole("textbox", { name: "UUID" }), { target: { value: "  " } });
  fireEvent.change(screen.getByRole("spinbutton", { name: "Port" }), { target: { value: "" } });
  await userEvent.click(screen.getByRole("button", { name: "添加协议" }));
  expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ host: "192.168.2.188", port: 8888, uuid: "A1B2C3D4" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it("opens running protocols read-only with an explicit home action", async () => {
  const onGoHome = vi.fn(), onSave = vi.fn(), onProbe = vi.fn();
  render(<MouseProtocolView {...base} running protocols={[protocol]} selectedId={protocol.id} onSave={onSave} onSelect={vi.fn()} onProbe={onProbe} onGoHome={onGoHome} />);
  await userEvent.click(screen.getByRole("button", { name: "配置协议" }));
  const dialog = within(screen.getByRole("dialog", { name: "查看协议" }));
  expect(dialog.getByText(/停止后可修改/)).toBeVisible();
  expect(dialog.getByRole("textbox", { name: "IP" })).toHaveValue(protocol.host);
  expect(dialog.getByRole("button", { name: "测试连通性" })).toBeDisabled();
  expect(dialog.queryByRole("button", { name: "保存并写入文件" })).not.toBeInTheDocument();
  await userEvent.click(dialog.getByRole("button", { name: "前往首页" }));
  expect(onGoHome).toHaveBeenCalledOnce();
  expect(onSave).not.toHaveBeenCalled();
  expect(onProbe).not.toHaveBeenCalled();
});

it("tests a new protocol using displayed defaults without saving", async () => {
  const onSave = vi.fn(), onSelect = vi.fn(), onProbe = vi.fn(async () => ({ elapsed_ms: 8 }));
  render(<MouseProtocolView {...base} onSave={onSave} onSelect={onSelect} onProbe={onProbe} />);
  await userEvent.click(screen.getByRole("button", { name: "新增协议" }));
  await userEvent.click(screen.getByRole("button", { name: "测试连通性" }));
  expect(await screen.findByText(/设备握手成功/)).toBeVisible();
  expect(onProbe).toHaveBeenCalledWith(expect.objectContaining({ model: "net_pro8k", host: "192.168.2.188", port: 8888, uuid: "A1B2C3D4" }));
  expect(onSave).not.toHaveBeenCalled();
  expect(onSelect).not.toHaveBeenCalled();
});

it("locates duplicates and requires confirmation to remove inactive protocols", async () => {
  const existing = { ...protocol, model: "net_pro8k" as const, host: "192.168.2.188", uuid: "a1b2c3d4" };
  const onSave = vi.fn(), onRemove = vi.fn(async () => {});
  render(<MouseProtocolView {...base} protocols={[existing]} onSave={onSave} onRemove={onRemove} onSelect={vi.fn()} onProbe={vi.fn()} />);
  await userEvent.click(screen.getByRole("button", { name: "新增协议" }));
  await userEvent.click(screen.getByRole("button", { name: "添加协议" }));
  expect(screen.getByRole("dialog", { name: "配置协议" })).toBeVisible();
  expect(onSave).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "删除协议" }));
  expect(onRemove).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "确认删除" }));
  expect(onRemove).toHaveBeenCalledWith(existing);
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
});

it("configures an inactive protocol without enabling it and protects unsaved edits when switching", async () => {
  const another = { ...protocol, id: "two", model: "nb" as const, host: "10.0.0.3" };
  const onSelect = vi.fn(), onSave = vi.fn(async () => {});
  render(<MouseProtocolView {...base} protocols={[protocol, another]} selectedId={another.id} onSave={onSave} onSelect={onSelect} onProbe={vi.fn()} />);
  await userEvent.click(screen.getByRole("button", { name: "配置协议" }));
  const picker = screen.getByRole("combobox", { name: "配置的协议" });
  expect(picker).toHaveValue(another.id);
  await userEvent.selectOptions(picker, protocol.id);
  expect(screen.getByRole("textbox", { name: "IP" })).toHaveValue(protocol.host);
  fireEvent.change(screen.getByRole("textbox", { name: "IP" }), { target: { value: "10.0.0.4" } });
  expect(screen.getByRole("combobox", { name: "配置的协议" })).toBeDisabled();
  await userEvent.click(screen.getByRole("button", { name: "保存并写入文件" }));
  expect(onSave).toHaveBeenCalledWith({ ...protocol, host: "10.0.0.4" });
  expect(onSelect).not.toHaveBeenCalled();
});

it("keeps activation evidence unchanged when card selection fails or a write is pending", async () => {
  const onSelect = vi.fn(async () => { throw new Error("write failed"); });
  const props = { ...base, protocols: [protocol], onSave: vi.fn(), onSelect, onProbe: vi.fn() };
  const view = render(<MouseProtocolView {...props} />);
  await userEvent.click(screen.getByRole("button", { name: /kmBox Net ·/ }));
  expect(screen.getByRole("button", { name: "空" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  view.rerender(<MouseProtocolView {...props} saving />);
  expect(screen.getByRole("button", { name: /kmBox Net ·/ })).toBeDisabled();
  expect(screen.getByRole("button", { name: "配置协议" })).toBeDisabled();
});
