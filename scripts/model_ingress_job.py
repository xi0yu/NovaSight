#!/usr/bin/env python3
"""Allowlisted offline TensorRT model-ingress worker for novasightd.

This process never owns the online runtime. Rust resolves the artifact path,
serializes lifecycle operations, applies time/output limits, and may terminate
the worker when stop or emergency-stop is requested.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any

# Source-tree deployments keep the package beside scripts/. Installed
# deployments still resolve the normal site-package first after this entry.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from novasight.inference.runtime import InferenceRuntime
from novasight.model_ingress import (
    EngineInspector,
    InferenceConfigBuilder,
    ModelProbe,
    ModelProfileResolver,
    ModelProfileStore,
    ModelStatus,
    PostprocessProfile,
    ValidationIssue,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NovaSight allowlisted model-ingress worker")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--engine", type=Path, required=True)
    inspect.add_argument("--display-name", required=True)

    configure = subparsers.add_parser("configure")
    configure.add_argument("--engine", type=Path, required=True)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--engine", type=Path, required=True)
    probe.add_argument("--parser-library", type=Path, required=True)
    probe.add_argument("--input-mode", choices=("fixed",), default="fixed")
    return parser.parse_args()


def inspect_engine(engine_path: Path, display_name: str) -> dict[str, Any]:
    inspection = EngineInspector().inspect(engine_path)
    profile = ModelProfileResolver().resolve(
        engine_path=engine_path,
        inspection=inspection,
        display_name=display_name,
    )
    profile_path = ModelProfileStore().write(profile)
    return _response(profile_path, profile)


def configure_engine(engine_path: Path) -> dict[str, Any]:
    payload = _read_request()
    store = ModelProfileStore()
    profile_path = store.existing_path_for_engine(engine_path)
    if not store.contains_model_profile(profile_path):
        raise ValueError("inspect the TensorRT engine first")
    profile = store.load(profile_path)
    if profile.status is ModelStatus.UNINSPECTED:
        raise ValueError("engine content changed; inspect it again before configuration")
    if profile.status is ModelStatus.ACTIVE:
        raise ValueError("active model semantics cannot be changed")
    configured = ModelProfileResolver().configure(
        profile,
        color_format=payload["color_format"],
        scale=payload["scale"],
        offsets=payload.get("offsets", []),
        mean=payload.get("mean", []),
        std=payload.get("std", []),
        resize_mode=payload["resize_mode"],
        symmetric_padding=payload.get("symmetric_padding", False),
        padding_value=payload.get("padding_value", 0.0),
        parser_type=payload["parser_type"],
        class_count=payload["class_count"],
        labels=payload["labels"],
        bbox_format=payload["bbox_format"],
        has_objectness=payload["has_objectness"],
    )
    configured = replace(
        configured,
        postprocess=PostprocessProfile(
            confidence_threshold=float(payload.get("confidence_threshold", 0.25)),
            nms_threshold=float(payload.get("nms_threshold", 0.45)),
            max_detections=int(payload.get("max_detections", 300)),
        ),
    )
    written_path = store.write(configured)
    if profile_path != written_path:
        profile_path.unlink(missing_ok=True)
    return _response(written_path, configured)


def probe_engine(engine_path: Path, parser_library: Path) -> dict[str, Any]:
    store = ModelProfileStore()
    profile_path = store.existing_path_for_engine(engine_path)
    if not store.contains_model_profile(profile_path):
        raise ValueError("inspect and configure the model first")
    profile = store.load(profile_path)
    if profile.status not in {ModelStatus.READY_FOR_PROBE, ModelStatus.VALIDATED}:
        raise ValueError(
            "model profile is not ready for diagnostics: " f"{profile.status.value}"
        )
    validated, report = ModelProbe().run(
        profile,
        inference=InferenceRuntime(),
    )
    runtime_manifest = None
    if validated.status is ModelStatus.VALIDATED:
        try:
            runtime_manifest = InferenceConfigBuilder().build(
                validated,
                parser_library_path=parser_library,
            ).manifest
        except ValueError as exc:
            validated = replace(
                validated,
                status=ModelStatus.INVALID,
                validation=replace(
                    validated.validation,
                    status="invalid",
                    issues=(*validated.validation.issues, "RUNTIME_CONFIG_INVALID"),
                ),
            )
            report = replace(
                report,
                status="invalid",
                issues=(
                    *report.issues,
                    ValidationIssue(
                        code="RUNTIME_CONFIG_INVALID",
                        stage="config",
                        message=str(exc),
                    ),
                ),
            )
    written_path = store.write(validated, runtime_manifest=runtime_manifest)
    if profile_path != written_path:
        profile_path.unlink(missing_ok=True)
    return _response(written_path, validated, report=report)


def _read_request() -> dict[str, Any]:
    raw = json.load(sys.stdin)
    if not isinstance(raw, dict):
        raise ValueError("model profile request must be a JSON object")
    return raw


def _response(
    profile_path: Path,
    profile: Any,
    *,
    report: Any | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "profile_path": str(profile_path.expanduser().resolve(strict=False)),
        "profile": _json_value(asdict(profile)),
    }
    if report is not None:
        response["report"] = _json_value(asdict(report))
    return response


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    try:
        if args.operation == "inspect":
            response = inspect_engine(args.engine, args.display_name)
        elif args.operation == "configure":
            response = configure_engine(args.engine)
        elif args.operation == "probe":
            response = probe_engine(args.engine, args.parser_library)
        else:  # pragma: no cover - argparse owns the allowlist.
            raise ValueError(f"unsupported operation: {args.operation}")
    except Exception as exc:
        print(
            json.dumps(
                {"code": "MODEL_INGRESS_JOB_FAILED", "message": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
