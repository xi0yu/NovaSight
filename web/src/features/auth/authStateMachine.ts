import type { AuthState } from "../../contracts/auth";

export type AuthAuthority = "client" | "server" | "user" | "user-confirmed";
const terminal = new Set<AuthState>(["required", "expired", "revoked", "signed_out"]);
export function transitionAuth(current: AuthState, next: AuthState, authority: AuthAuthority): AuthState {
  if (current === next) return current;
  if (next === "signed_out") return authority === "user" ? next : current;
  if (terminal.has(next)) return authority === "server" ? next : current;
  if (next === "authorized") {
    if (terminal.has(current)) return authority === "user-confirmed" ? next : current;
    return authority === "server" || authority === "user-confirmed" ? next : current;
  }
  if (terminal.has(current)) return next === "checking" && authority === "user" ? next : current;
  return next;
}
