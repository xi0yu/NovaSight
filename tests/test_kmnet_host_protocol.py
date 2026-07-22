from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _request(process: subprocess.Popen[str], payload: dict[str, object]) -> dict[str, object]:
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()
    response = process.stdout.readline()
    assert response, process.stderr.read() if process.stderr is not None else "helper exited"
    return json.loads(response)


def test_helper_loads_vendor_boundary_and_serves_narrow_protocol(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    (tmp_path / "kmNet.py").write_text(
        """
import json, os
def _record(name, args):
    with open(os.environ['KMNET_TEST_LOG'], 'a', encoding='utf-8') as handle:
        handle.write(json.dumps([name, list(args)]) + '\\n')
def init(*args): _record('init', args); return 0
def monitor(*args): _record('monitor', args); return None
def move(*args): _record('move', args); return 0
def isdown_left(): return 1
def isdown_right(): return 0
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["KMNET_TEST_LOG"] = str(log)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path.cwd())])
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "novasight.executors.kmnet_host"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        hello = _request(process, {"id": 1, "op": "hello", "protocol": 1})
        connected = _request(
            process,
            {
                "id": 2,
                "op": "connect",
                "host": "192.0.2.10",
                "port": 8888,
                "uuid": "box",
                "monitor_port": 5001,
            },
        )
        moved = _request(process, {"id": 3, "op": "move", "dx": 120, "dy": -45})
        buttons = _request(process, {"id": 4, "op": "buttons"})
        stopped = _request(process, {"id": 5, "op": "shutdown"})
        assert hello["ok"] is True
        assert hello["result"]["protocol"] == 1
        assert connected["ok"] is True
        assert moved["ok"] is True
        assert buttons["result"] == {"available": True, "left": True, "right": False}
        assert stopped["ok"] is True
        assert process.wait(timeout=2) == 0
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert calls == [
            ["init", ["192.0.2.10", "8888", "box"]],
            ["monitor", [5001]],
            ["move", [120, -45]],
        ]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)


def test_helper_rejects_bad_driver_return_code(tmp_path: Path) -> None:
    (tmp_path / "kmNet.py").write_text(
        "def init(*args): return 0\ndef move(*args): return 9\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path.cwd())])
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "novasight.executors.kmnet_host"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert _request(process, {"id": 1, "op": "hello", "protocol": 1})["ok"] is True
        assert _request(
            process,
            {
                "id": 2,
                "op": "connect",
                "host": "192.0.2.10",
                "port": 8888,
                "uuid": "box",
                "monitor_port": 0,
            },
        )["ok"] is True
        response = _request(process, {"id": 3, "op": "move", "dx": 1, "dy": 2})
        assert response["ok"] is False
        assert "move failed rc=9" in response["error"]
    finally:
        process.kill()
        process.wait(timeout=2)
