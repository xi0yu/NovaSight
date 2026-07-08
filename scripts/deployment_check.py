#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: dict[str, Any]


@dataclass(frozen=True)
class DeploymentCheckSummary:
    unit_path: str
    base_url: str
    checks: list[CheckResult]
    passed: bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate NovaSight Phase 9 deployment prerequisites.")
    parser.add_argument("--unit", default="deploy/novasight.service")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--header", action="append", default=[], help="Extra HTTP header as Name: Value.")
    parser.add_argument("--skip-api", action="store_true", help="Only check the systemd unit file.")
    args = parser.parse_args()

    summary = run_deployment_check(
        unit_path=Path(args.unit),
        base_url=args.base_url,
        headers=_parse_headers(args.header),
        skip_api=args.skip_api,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.passed else 2


def run_deployment_check(
    *,
    unit_path: Path,
    base_url: str,
    headers: dict[str, str],
    skip_api: bool,
) -> DeploymentCheckSummary:
    unit = _read_unit(unit_path)
    checks = [
        _unit_check(unit, "Service", "Type", "notify"),
        _unit_check(unit, "Service", "WatchdogSec", "10"),
        _unit_check(unit, "Service", "Restart", "on-failure"),
        _unit_check(unit, "Service", "StandardOutput", "journal"),
        _unit_check(unit, "Service", "StandardError", "journal"),
        _unit_check(unit, "Service", "RuntimeDirectory", "novasight"),
        _unit_check(unit, "Service", "LogsDirectory", "novasight"),
        _environment_check(
            unit,
            expected="NOVASIGHT_INSTANCE_LOCK=/run/novasight/instance.lock",
        ),
    ]
    if not skip_api:
        checks.extend(_api_checks(base_url=base_url, headers=headers))
    return DeploymentCheckSummary(
        unit_path=str(unit_path),
        base_url=base_url,
        checks=checks,
        passed=all(check.passed for check in checks),
    )


def _read_unit(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False)
    parser.optionxform = str  # type: ignore[method-assign]
    with path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    return parser


def _unit_check(
    unit: configparser.ConfigParser,
    section: str,
    key: str,
    expected: str,
) -> CheckResult:
    actual = unit.get(section, key, fallback="")
    return CheckResult(
        name=f"unit.{section}.{key}",
        passed=actual == expected,
        detail={
            "actual": actual,
            "expected": expected,
        },
    )


def _environment_check(unit: configparser.ConfigParser, *, expected: str) -> CheckResult:
    values = unit.get("Service", "Environment", fallback="")
    actual = values if isinstance(values, str) else ""
    return CheckResult(
        name="unit.Service.Environment.instance_lock",
        passed=expected in actual,
        detail={
            "actual": actual,
            "expected_contains": expected,
        },
    )


def _api_checks(*, base_url: str, headers: dict[str, str]) -> list[CheckResult]:
    health = _request_json(base_url, "/healthz", headers=headers)
    system = _request_json(base_url, "/api/v1/system", headers=headers)
    system_body = system.get("body") if isinstance(system.get("body"), dict) else {}
    instance_lock = system_body.get("instance_lock") if isinstance(system_body, dict) else {}
    process = system_body.get("process") if isinstance(system_body, dict) else {}
    return [
        CheckResult(
            name="api.healthz",
            passed=health.get("status_code") == 200 and bool((health.get("body") or {}).get("ok")),
            detail=health,
        ),
        CheckResult(
            name="api.system.instance_lock",
            passed=(
                system.get("status_code") == 200
                and isinstance(instance_lock, dict)
                and instance_lock.get("configured") is True
                and instance_lock.get("acquired") is True
                and instance_lock.get("path") == "/run/novasight/instance.lock"
            ),
            detail={
                "status_code": system.get("status_code"),
                "instance_lock": instance_lock,
            },
        ),
        CheckResult(
            name="api.system.process_memory",
            passed=(
                system.get("status_code") == 200
                and isinstance(process, dict)
                and float(process.get("rss_mb", 0.0)) > 0.0
            ),
            detail={
                "status_code": system.get("status_code"),
                "process": process,
            },
        ),
    ]


def _request_json(base_url: str, path: str, *, headers: dict[str, str]) -> dict[str, Any]:
    request = Request(f"{base_url.rstrip('/')}{path}", headers=headers)
    try:
        with urlopen(request, timeout=5.0) as response:
            return {
                "status_code": int(response.status),
                "body": _decode_json(response.read()),
            }
    except HTTPError as exc:
        return {
            "status_code": int(exc.code),
            "body": _decode_json(exc.read()),
        }
    except (TimeoutError, URLError) as exc:
        return {
            "status_code": 0,
            "body": {"detail": str(exc)},
        }


def _decode_json(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        name, sep, header_value = value.partition(":")
        if sep and name.strip():
            headers[name.strip()] = header_value.strip()
    return headers


if __name__ == "__main__":
    raise SystemExit(main())
