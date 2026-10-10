import { describe, expect, it } from "vitest";

import { decodeCaptureDevices } from "./capture";

describe("capture contract", () => {
  it("decodes discovered devices without inventing a default", () => {
    expect(decodeCaptureDevices({ devices: ["/dev/video2", "/dev/video0", "/dev/video2"] })).toEqual({
      devices: ["/dev/video2", "/dev/video0"],
    });
    expect(decodeCaptureDevices({ devices: [] })).toEqual({ devices: [] });
  });

  it("rejects blank device paths", () => {
    expect(() => decodeCaptureDevices({ devices: [" "] })).toThrow(/non-empty string/);
  });
});
