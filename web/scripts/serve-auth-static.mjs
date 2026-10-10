import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";
const directory = mkdtempSync(join(tmpdir(), "ns-auth-static-"));
writeFileSync(join(directory, "config.yaml"), `server:\n  host: 127.0.0.1\n  port: 7352\n  control_socket: ${directory}/daemon.sock\npaths:\n  license: ${directory}/license.json\n`);
const child = spawn(resolve("../out/cargo/debug/novasight-web"), ["--config", join(directory, "config.yaml")], {
  cwd: directory, stdio: "inherit", env: { ...process.env, NOVASIGHT_WEB_ROOT: resolve("../out/web") },
});
for (const signal of ["SIGTERM", "SIGINT"]) process.on(signal, () => child.kill(signal));
child.on("exit", (code) => { rmSync(directory, { recursive: true, force: true }); process.exitCode = code ?? 1; });
