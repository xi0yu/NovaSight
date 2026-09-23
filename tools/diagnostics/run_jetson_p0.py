#!/usr/bin/env python3
"""Run an existing development daemon in an isolated, output-disabled directory.

Run on Jetson after compiling pipeline_probe.cpp. Never installs software,
changes the source checkout, activates a deployment, or connects a pointer.
"""
import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import sqlite3
import subprocess
import time


def require_closed(snapshot):
    metrics = snapshot.get("pipeline_metrics")
    if metrics is None:
        raise RuntimeError("P0 snapshot has no pipeline_metrics; cannot verify output closure")
    # UncommissionedPointerDevice.connect() is a no-op returning Ok; the runtime
    # therefore reports connected=True even with this inert adapter. Config is
    # checked separately before start; these are the actual output invariants.
    required = {"output_gate_open": False, "device_receipts": 0}
    for key, expected in required.items():
        if type(metrics.get(key)) is not type(expected) or metrics[key] != expected:
            raise RuntimeError(f"P0_OUTPUT_NOT_CLOSED: {key}={metrics.get(key)!r}")


def self_test():
    fields = {"output_gate_open": False, "device_receipts": 0}
    require_closed({"pipeline_metrics": fields})
    for name in fields:
        bad = dict(fields, **{name: 1 if name == "device_receipts" else True})
        try:
            require_closed({"pipeline_metrics": bad})
        except RuntimeError:
            continue
        raise AssertionError(name)
    try:
        require_closed({})
    except RuntimeError:
        print("P0_RUNNER_SELF_TEST_PASS")
        return
    raise AssertionError("missing telemetry was accepted")


def run(args):
    import yaml  # Already installed on the target; not a product dependency.
    repo, stage = args.repo.resolve(), args.stage.resolve()
    if stage.parent != Path("/tmp") or not stage.name.startswith("novasight-p0-"):
        raise RuntimeError("P0 stage must be a dedicated /tmp/novasight-p0-* directory")
    if not 1 <= args.seconds <= 120:
        raise RuntimeError("P0 run duration must be 1..120 seconds")
    if args.profile_cuda and not args.no_probe:
        raise RuntimeError("Use --no-probe for the separate Nsight run")
    if args.fault_gpu_after and (args.no_probe or args.profile_cuda):
        raise RuntimeError("GPU fault injection requires the diagnostic probe without Nsight")
    used = subprocess.run(["fuser", "/dev/video0"], capture_output=True, text=True)
    if used.returncode == 0:
        raise RuntimeError("P0 capture device is already in use")
    if used.returncode != 1:
        raise RuntimeError("P0 could not determine capture device ownership")
    work = stage / f"run-{time.time_ns()}"
    work.mkdir(mode=0o700)
    for name in ("data", "run", "logs"):
        (work / name).mkdir(mode=0o700)
    cfg = yaml.safe_load((repo / args.config).read_text())
    if args.capture_fps:
        cfg["capture"]["fps"] = args.capture_fps
        cfg["capture"]["preference"] = "manual"
    cfg["control"]["output_enabled"] = False
    cfg["hardware"]["auto_connect"] = False
    cfg.setdefault("server", {})["control_socket"] = str(work / "run/control.sock")
    cfg["paths"] = {"data_dir": str(work / "data"), "model_dir": str(repo / "data/models"),
                    "database": str(work / "data/catalog.db"), "license": str(work / "data/license.json")}
    ini = configparser.ConfigParser()
    ini.read(repo / "data/runtime/deepstream/active-nvinfer.ini")
    original = Path(ini["property"]["custom-lib-path"])
    probe = stage / "libpipeline_probe.so"
    if not original.is_file() or (not args.no_probe and not probe.is_file()):
        raise RuntimeError("P0 parser/probe library is missing")
    cfg["inference"]["deepstream_parser_library"] = str(original if args.no_probe else probe)
    config_path = work / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    source = sqlite3.connect(f"file:{repo}/data/novasight.db?mode=ro", uri=True)
    copied = sqlite3.connect(work / "data/catalog.db")
    source.backup(copied)
    source.close()
    copied.close()
    binary_dir = args.binary_dir.resolve() if args.binary_dir else repo / "out/cargo/debug"
    daemon = binary_dir / "novasightd"
    ctl = binary_dir / "novasightctl"
    code = secrets.token_hex(24)
    code_path = work / "run/development-access"
    code_path.write_text(code)
    code_path.chmod(0o600)
    env = dict(os.environ, NOVASIGHT_TEMPORARY_LICENSE_CODE=code,
               NOVASIGHT_P0_ORIGINAL_PARSER=str(original),
               NOVASIGHT_P0_TRACE=str(work / "trace.csv"))
    if not args.no_probe:
        env["LD_PRELOAD"] = str(probe)
    if args.fault_gpu_after:
        env["NOVASIGHT_P0_FAIL_GPU_AFTER"] = str(args.fault_gpu_after)
    client_env = dict(os.environ, NOVASIGHT_CONTROL_SOCKET=str(work / "run/control.sock"))
    client_env.pop("LD_PRELOAD", None)

    def call(*command, timeout=20):
        result = subprocess.run([str(ctl), *command], cwd=work, env=client_env,
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"P0 command {command[0]} failed: {result.stderr[-1500:]}")
        response = json.loads(result.stdout)
        if command[0] in {"status", "start", "stop"}:
            (work / "last-response.json").write_text(json.dumps(response, indent=2))
        return response

    summary = {"directory": str(work), "instrumented": not args.no_probe,
               "profile_cuda": args.profile_cuda,
               "config_source": args.config, "capture": cfg["capture"],
               "seconds_requested": args.seconds, "binary": str(daemon),
               "binary_sha256": hashlib.sha256(daemon.read_bytes()).hexdigest(),
               "checkout_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
               "fault_gpu_after": args.fault_gpu_after}
    print(json.dumps({"event": "P0_ISOLATED_START", **summary}), flush=True)
    command = [str(daemon), "--config", str(config_path)]
    if args.profile_cuda:
        command = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
                   "--cpuctxsw=none", "--output", str(work / "cuda"), *command]
    with (work / "logs/daemon.log").open("w") as log, (work / "tegrastats.log").open("w") as stats:
        monitor = None
        process = subprocess.Popen(command, cwd=work,
                                   env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            monitor = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=stats,
                                       stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 30
            while not (work / "run/control.sock").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("P0 daemon did not become ready")
                time.sleep(0.1)
            call("license", "activate", "--key-file", str(code_path))
            code_path.unlink()
            effective = call("config", "show")
            if effective["control"]["output_enabled"] is not False or effective["hardware"]["auto_connect"] is not False:
                raise RuntimeError("P0 effective config does not close physical output")
            require_closed(call("status"))
            started = call("start")
            require_closed(started)
            if started["pipeline"]["state"] != "running":
                raise RuntimeError("P0 pipeline did not enter running state")
            print(json.dumps({"event": "P0_MEASURING", "pid": process.pid}), flush=True)
            deadline = time.monotonic() + args.seconds
            with (work / "snapshots.jsonl").open("w") as snapshots:
                while time.monotonic() < deadline:
                    snapshot = call("status")
                    require_closed(snapshot)
                    snapshots.write(json.dumps({"observed_ns": time.monotonic_ns(), "snapshot": snapshot}) + "\n")
                    snapshots.flush()
                    if snapshot["pipeline"]["state"] != "running":
                        error = snapshot["pipeline"].get("last_error") or {}
                        if (args.fault_gpu_after and snapshot["pipeline"]["state"] == "faulted"
                                and "Strict GPU frame failed" in error.get("message", "")):
                            summary["expected_gpu_fault"] = True
                            break
                        raise RuntimeError("P0 runtime stopped during measurement")
                    time.sleep(1)
            summary["final_snapshot"] = snapshot
            if args.fault_gpu_after and not summary.get("expected_gpu_fault"):
                raise RuntimeError("Requested GPU fault was not observed")
            stopped = call("stop")
            require_closed(stopped)
            summary["stop_state"] = stopped["pipeline"]["state"]
            summary["result"] = "PASS"
        except Exception as error:
            summary["result"] = "FAIL"
            summary["error"] = str(error)
        finally:
            if process.poll() is None:
                try:
                    call("shutdown", timeout=12)
                except Exception:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=45 if args.profile_cuda else 12)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            code_path.unlink(missing_ok=True)
            summary["daemon_exit"] = process.returncode
            if monitor is not None:
                monitor.terminate()
                monitor.wait(timeout=5)
    if summary.get("result") == "PASS" and summary["daemon_exit"] != 0:
        summary["result"] = "MEASURED_WITH_SHUTDOWN_ERROR"
    (work / "receipt.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    display = {k: v for k, v in summary.items() if k != "final_snapshot"}
    if summary["result"] != "PASS":
        display["log_tail"] = (work / "logs/daemon.log").read_text(errors="replace")[-7000:]
    print(json.dumps(display, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["result"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--binary-dir", type=Path, help="Use isolated build artifacts without replacing deployed binaries")
    parser.add_argument("--capture-fps", type=int, choices=[120, 240], help="Temporary capture override; resolves the device fraction")
    parser.add_argument("--fault-gpu-after", type=int, choices=range(100, 2001), metavar="100..2000")
    parser.add_argument("--config", default="data/novasight.yaml",
                        choices=["data/novasight.yaml", "data/novasight.frontend-dev.yaml"])
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--profile-cuda", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    options = parser.parse_args()
    if options.self_test:
        self_test()
    elif options.repo is None or options.stage is None:
        parser.error("--repo and --stage are required")
    else:
        raise SystemExit(run(options))
