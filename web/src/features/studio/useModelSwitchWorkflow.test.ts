import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { publishModel, rollbackModel, type ModelCatalogModel, type ModelPublishResponse } from "../../api";
import type { ActionConfirmationRequest } from "./ActionConfirmationDialog";
import { isActiveModelNoOp, useModelSwitchWorkflow } from "./useModelSwitchWorkflow";

function response(overrides: Partial<ModelPublishResponse["report"]> = {}): ModelPublishResponse {
  return {
    deployment: { artifact_id: 42 },
    report: {
      applied: false,
      rolled_back: false,
      artifact_id: 42,
      ...overrides,
    },
  } as unknown as ModelPublishResponse;
}

describe("model switch no-op", () => {
  it("accepts selecting the already active artifact as a no-op success", () => {
    expect(isActiveModelNoOp(response(), 42)).toBe(true);
  });

  it("does not hide rollback or mismatched-artifact failures", () => {
    expect(isActiveModelNoOp(response({ rolled_back: true }), 42)).toBe(false);
    expect(isActiveModelNoOp(response({ artifact_id: 41 }), 42)).toBe(false);
    expect(isActiveModelNoOp(response(), 41)).toBe(false);
  });
});

afterEach(() => vi.unstubAllGlobals());

it("asks before registering and deploying a model while the mainline is stopped", () => {
  const fetchMock = vi.fn(() => new Promise<Response>(() => {}));
  vi.stubGlobal("fetch", fetchMock);
  const setConfirmationRequest = vi.fn();
  const model = {
    type: "model", kind: "engine", name: "candidate.engine", relative_path: "candidate.engine",
    size_bytes: 100, scan_status: "need_confirm", scan_reason: "", recommendation: "unrated", tags: [],
  } as ModelCatalogModel;
  const { result } = renderHook(() => useModelSwitchWorkflow({
    applyModelCatalogResult: vi.fn(), onRefresh: vi.fn(async () => {}), parserPreset: "auto",
    physicalOutputEnabled: false, runtimeMainlineRunning: false, selectedCatalogModel: model,
    setBusy: vi.fn(), setConfirmationRequest, setLocalError: vi.fn(),
    setModelCatalogMessage: vi.fn(), setModelCatalogRefreshKey: vi.fn(),
    setModelDetailsRefreshKey: vi.fn(), setModelManagerDialogOpen: vi.fn(),
    setSelectedModelArtifactId: vi.fn(), setSelectedModelProjectId: vi.fn(),
    setSelectedModelVersionId: vi.fn(),
  }));

  act(() => result.current.switchModel());
  expect(setConfirmationRequest).toHaveBeenCalledOnce();
  const request = setConfirmationRequest.mock.calls[0][0] as ActionConfirmationRequest;
  expect(request.details?.join(" ")).toContain("登记");
  expect(request.description).toContain("当前模型");
  expect(fetchMock).not.toHaveBeenCalled();
});

it("sends physical-output acknowledgement only when explicitly requested", async () => {
  const fetchMock = vi.fn(async (_url: RequestInfo | URL, _init?: RequestInit) => new Response("blocked", { status: 428 }));
  vi.stubGlobal("fetch", fetchMock);
  await expect(publishModel(1, 2)).rejects.toThrow();
  await expect(publishModel(1, 2, "auto", true)).rejects.toThrow();
  await expect(rollbackModel(1)).rejects.toThrow();
  await expect(rollbackModel(1, true)).rejects.toThrow();
  expect(fetchMock.mock.calls.map(([, init]) => new Headers(init?.headers).get("x-novasight-physical-output-ack")))
    .toEqual([null, "confirmed", null, "confirmed"]);
});
