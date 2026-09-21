#!/usr/bin/env python3
"""Exercise class-profile save/delete through real isolated daemon and Web API."""

import copy
import http.cookiejar
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DAEMON = ROOT / "out/cargo/debug/novasightd"
WEB = ROOT / "out/cargo/debug/novasight-web"


def main():
    with tempfile.TemporaryDirectory(prefix="novasight-class-config-") as temporary:
        workspace = Path(temporary)
        (workspace / "data/models").mkdir(parents=True)
        (workspace / "run").mkdir()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        bootstrap = (ROOT / "crates/novasight-config/src/bootstrap.yaml").read_text()
        assert "  port: 7351" in bootstrap and "  host: 0.0.0.0" in bootstrap
        config_path = workspace / "data/novasight.yaml"
        config_path.write_text(
            bootstrap.replace("  port: 7351", f"  port: {port}", 1)
            .replace("  host: 0.0.0.0", "  host: 127.0.0.1", 1)
        )
        code = secrets.token_hex(32)
        env = {**os.environ, "NOVASIGHT_TEMPORARY_LICENSE_CODE": code}
        processes = []
        try:
            for command, process_env in (
                ([str(DAEMON), "--config", str(config_path)], env),
                ([str(WEB), "--config", str(config_path)], {**os.environ, "NOVASIGHT_WEB_ROOT": str(ROOT / "out/web")}),
            ):
                processes.append(subprocess.Popen(command, cwd=workspace, env=process_env,
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            base = f"http://127.0.0.1:{port}"
            opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
            )

            def request(path, payload=None, csrf=None):
                headers = {"Content-Type": "application/json"} if payload is not None else {}
                if csrf:
                    headers["x-novasight-csrf"] = csrf
                data = json.dumps(payload).encode() if payload is not None else None
                try:
                    response = opener.open(urllib.request.Request(base + path, data=data, headers=headers), timeout=10)
                except urllib.error.HTTPError as error:
                    raise AssertionError(f"{path} HTTP {error.code}: {error.read().decode()[:500]}") from error
                with response:
                    return json.load(response)

            for _ in range(100):
                if any(process.poll() is not None for process in processes):
                    raise AssertionError("isolated daemon or Web API exited before readiness")
                try:
                    request("/healthz")
                    break
                except urllib.error.URLError:
                    time.sleep(0.1)
            else:
                raise AssertionError("isolated Web API did not become ready")

            csrf = request("/api/auth/session", {"key": code})["csrf_token"]
            current = request("/api/config")
            assert current["control"]["output_enabled"] is False
            assert current["hardware"]["auto_connect"] is False
            assert "detection_class_profiles" not in current["inference"]
            candidate = copy.deepcopy(current)
            candidate["inference"].update({
                "detection_class_profile": "default",
                "detection_class_profiles": {"default": ["enemy"], "obsolete": ["enemy"]},
                "detection_class_priorities": {"default": "0", "obsolete": "0"},
                "detection_class_filters": {"default": "all", "obsolete": "all"},
            })
            candidate["control"]["aim"] = {
                "class_roles": {"default": {"0": "head"}, "obsolete": {"0": "body"}},
                "role_y_ratios": {"head": 0.33, "body": 0.5, "other": 0.22},
            }
            candidate["pipeline"]["target_class_aim_y_ratios"] = "0:0.33"
            result = request("/api/config", candidate, csrf)
            assert result["apply_mode"] == "epoch_reload" and result["applied"] is True, result
            saved = request("/api/config")
            assert saved["inference"]["detection_class_profiles"]["obsolete"] == ["enemy"]
            assert saved["control"]["aim"]["class_roles"]["obsolete"]["0"] == "body"

            deleted = copy.deepcopy(saved)
            for key in ("detection_class_profiles", "detection_class_priorities", "detection_class_filters"):
                deleted["inference"][key].pop("obsolete")
            deleted["control"]["aim"]["class_roles"].pop("obsolete")
            result = request("/api/config", deleted, csrf)
            assert result["apply_mode"] == "epoch_reload" and result["applied"] is True, result
            final = request("/api/config")
            assert all("obsolete" not in final["inference"][key] for key in (
                "detection_class_profiles", "detection_class_priorities", "detection_class_filters"
            ))
            assert "obsolete" not in final["control"]["aim"]["class_roles"]
            assert final["pipeline"]["target_class_aim_y_ratios"] == "0:0.33"
            assert final["control"]["output_enabled"] is False
            print("CLASS_CONFIG_HTTP_PASS login=real save=epoch_reload readback=verified delete=verified output=off")
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
            for process in reversed(processes):
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"CLASS_CONFIG_HTTP_FAIL {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
