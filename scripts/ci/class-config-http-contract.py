#!/usr/bin/env python3
"""Exercise class-profile and catalog edits through real isolated daemon and Web API."""

import copy
import http.cookiejar
import json
import os
import re
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
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "crates/novasight-config/src/bootstrap.yaml"
    if not source.is_file():
        raise ValueError(f"configuration source does not exist: {source}")
    with tempfile.TemporaryDirectory(prefix="novasight-class-config-") as temporary:
        workspace = Path(temporary)
        (workspace / "data/models").mkdir(parents=True)
        (workspace / "run").mkdir()
        (workspace / "run").chmod(0o700)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        source_text = source.read_text()
        for pattern, replacement in (
            (r"(?m)^  host: [^\n]+$", "  host: 127.0.0.1"),
            (r"(?m)^  port: \d+$", f"  port: {port}"),
            (r"(?m)^  output_enabled: (?:true|false)$", "  output_enabled: false"),
            (r"(?m)^  auto_connect: (?:true|false)$", "  auto_connect: false"),
        ):
            source_text, count = re.subn(pattern, replacement, source_text, count=1)
            if count != 1:
                raise AssertionError(f"source configuration lacks safe test field: {pattern}")
        for field in ("control_socket", "data_dir", "model_dir", "database", "license"):
            match = re.search(rf"(?m)^  {field}: ([^\n]+)$", source_text)
            if not match or Path(match.group(1)).is_absolute() or ".." in Path(match.group(1)).parts:
                raise AssertionError(f"source configuration has an unsafe test path: {field}")
        config_path = workspace / "data/novasight.yaml"
        config_path.write_text(source_text)
        code = secrets.token_hex(32)
        env = {**os.environ, "NOVASIGHT_TEMPORARY_LICENSE_CODE": code}
        processes = []
        log_files = []
        try:
            for name, command, process_env in (
                ("daemon", [str(DAEMON), "--config", str(config_path)], env),
                ("web", [str(WEB), "--config", str(config_path)], {**os.environ, "NOVASIGHT_WEB_ROOT": str(ROOT / "out/web")}),
            ):
                log_files.append((name, (workspace / f"{name}.log").open("w")))
                processes.append(subprocess.Popen(command, cwd=workspace, env=process_env,
                                                  stdout=log_files[-1][1], stderr=subprocess.STDOUT))
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
                    failures = [
                        f"{name} exit={process.returncode}: {(workspace / f'{name}.log').read_text()[-1000:]}"
                        for (name, _), process in zip(log_files, processes)
                        if process.poll() is not None
                    ]
                    raise AssertionError("isolated service exited before readiness: " + " | ".join(failures))
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
            runtime = request("/api/runtime/state")
            assert runtime["config"]["version"] == final["revision"], runtime["config"]
            assert runtime["config"]["effective_version"] == final["revision"], runtime["config"]
            assert runtime["config"]["restart_required"] is False, runtime["config"]
            assert runtime["vision"]["control"]["will_emit"] is not True
            model_root = workspace / "data/models"
            (model_root / "catalog-smoke.engine").write_bytes(b"catalog-only-not-a-loadable-engine")
            assert request("/api/models/catalog/folders", {"relative_path": "Arena"}, csrf)["relative_path"] == "Arena"
            assert request("/api/models/catalog/move", {
                "from_path": "catalog-smoke.engine", "to_path": "Arena/renamed.engine",
            }, csrf)["relative_path"] == "Arena/renamed.engine"
            catalog = request("/api/models/catalog?force=true")
            arena = next(node for node in catalog["root"]["children"] if node["type"] == "directory" and node["relative_path"] == "Arena")
            assert [node["relative_path"] for node in arena["children"]] == ["Arena/renamed.engine"], catalog
            assert not (model_root / "catalog-smoke.engine").exists()
            assert (model_root / "Arena/renamed.engine").read_bytes() == b"catalog-only-not-a-loadable-engine"
            print("CLASS_CONFIG_HTTP_PASS login=real save=epoch_reload readback=verified delete=verified effective=verified output=off")
            print("MODEL_CATALOG_HTTP_PASS folder=create move=confirmed readback=verified output=off")
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
            for _, log_file in log_files:
                log_file.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"CLASS_CONFIG_HTTP_FAIL {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
