import { describe, expect, it } from "vitest";

import { decodeActivityClear, decodeActivityEventFrame, decodeActivityHistory } from "./activity";

const event = {
  daemon_instance_id: "daemon-1",
  id: 7,
  occurred_at_ms: 1234,
  level: "error",
  tag: "参数",
  title: "操作失败",
  message: "参数没有生效",
  technical_detail: "validation failed",
  request_id: "request-1",
  count: 1,
};

describe("activity contract", () => {
  it("decodes history and realtime frames with the same event contract", () => {
    expect(decodeActivityHistory({ events: [event] }).events).toEqual([event]);
    expect(decodeActivityEventFrame({ kind: "activity_event", event }).event).toEqual(event);
  });

  it("rejects malformed severity and duplicate count values", () => {
    expect(() => decodeActivityHistory({ events: [{ ...event, level: "fatal" }] })).toThrow(/level/);
    expect(() => decodeActivityHistory({ events: [{ ...event, count: 0 }] })).toThrow(/count/);
  });

  it("decodes a cleared activity count", () => {
    expect(decodeActivityClear({ cleared: 4 })).toEqual({ cleared: 4 });
    expect(() => decodeActivityClear({ cleared: -1 })).toThrow(/cleared/);
  });
});
