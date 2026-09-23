import { describe, expect, it } from "vitest";

import { activateClassPolicy } from "./targetClassPolicy";

describe("class policy activation", () => {
  it("flattens the selected internal policy without changing raw cls", () => {
    const active = activateClassPolicy(
      ["alpha", "beta"],
      { beta: { "0": "body", "1": "head" } },
      { alpha: "0,1", beta: "1,0" },
      { alpha: "all", beta: "1" },
      "beta",
      "0,1",
      "all",
      { head: 0.2, body: 0.5, other: 0.4 }
    );

    expect(active.priority).toBe("1,0");
    expect(active.filter).toBe("1");
    expect(active.aimYRatios).toBe("0:0.50,1:0.20");
  });
});
