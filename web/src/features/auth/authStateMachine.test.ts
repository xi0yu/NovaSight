import { describe, expect, it } from "vitest";
import { transitionAuth } from "./authStateMachine";

describe("frozen transition authority", () => {
  it("rejects every client-created terminal and accepts a server-confirmed one", () => {
    for (const current of ["authorized", "checking", "reconnecting", "refreshing"] as const) {
      for (const terminal of ["required", "expired", "revoked"] as const) {
        expect(transitionAuth(current, terminal, "client")).toBe(current);
        expect(transitionAuth(current, terminal, "server")).toBe(terminal);
      }
    }
  });
  it("requires user intent to leave a terminal, and server confirmation to authorize", () => {
    for (const terminal of ["required", "expired", "revoked", "signed_out"] as const) {
      expect(transitionAuth(terminal, "authorized", "server")).toBe(terminal);
      expect(transitionAuth(terminal, "checking", "client")).toBe(terminal);
      expect(transitionAuth(terminal, "authorized", "user-confirmed")).toBe("authorized");
    }
    expect(transitionAuth("authorized", "signed_out", "client")).toBe("authorized");
    expect(transitionAuth("authorized", "signed_out", "user")).toBe("signed_out");
    expect(transitionAuth("checking", "authorized", "client")).toBe("checking");
  });
});
