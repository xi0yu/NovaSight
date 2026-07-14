from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import time
from typing import Any, ContextManager

from novasight.model_ingress import (
    ModelProfile,
    ModelStatus,
    ModelValidationReport,
    ProfileValidation,
    ValidationIssue,
    model_profile_validation_fingerprint,
)

from .runtime_pipeline import create_deepstream_probe_backend


def probe_deepstream_model(
    profile: ModelProfile,
    *,
    runtime: Any,
    isolation: ContextManager[Any] | None = None,
    timeout_s: float = 30.0,
) -> tuple[ModelProfile, ModelValidationReport]:
    """Validate a candidate with the same nvinfer/parser path used in production."""

    if profile.status not in {ModelStatus.READY_FOR_PROBE, ModelStatus.VALIDATED}:
        raise ValueError(
            "model profile must be READY_FOR_PROBE before DeepStream diagnostics "
            f"(got {profile.status.value})"
        )
    context = isolation if isolation is not None else nullcontext()
    report: ModelValidationReport
    try:
        with context:
            backend = create_deepstream_probe_backend(runtime=runtime, profile=profile)
            try:
                backend.start()
                report = _wait_for_detection_batch(
                    backend,
                    timeout_s=max(1.0, float(timeout_s)),
                )
            finally:
                backend.stop()
    except Exception as exc:
        report = _failure_report(
            code="DEEPSTREAM_PROBE_FAILED",
            stage="deepstream_startup",
            message=str(exc),
        )

    passed = report.status == "validated"
    validated = replace(
        profile,
        status=ModelStatus.VALIDATED if passed else ModelStatus.INVALID,
        validation=ProfileValidation(
            status=report.status,
            validated_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            engine_execution_ok=report.engine_execution_ok,
            decoder_ok=report.decoder_ok,
            nms_ok=report.nms_ok,
            detection_batch_ok=report.detection_batch_ok,
            profile_fingerprint=(
                model_profile_validation_fingerprint(profile) if passed else ""
            ),
            issues=tuple(issue.code for issue in report.issues),
        ),
    )
    return validated, report


def _wait_for_detection_batch(backend: Any, *, timeout_s: float) -> ModelValidationReport:
    deadline = time.monotonic() + timeout_s
    status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        batch = backend.detection_batch_mailbox.acquire_latest(
            after_generation=-1,
            timeout_s=min(0.1, remaining),
        )
        status = dict(backend.status())
        parser = _mapping(status.get("parser"))
        if int(parser.get("parse_failures") or 0) > 0:
            return _failure_report(
                code="DEEPSTREAM_PARSER_FAILED",
                stage="decode",
                message=(
                    "DeepStream C++ parser rejected the model output "
                    f"(error_code={int(parser.get('last_error_code') or 0)})"
                ),
                status=status,
            )
        if status.get("terminal_error") is True:
            return _failure_report(
                code="DEEPSTREAM_PIPELINE_FAILED",
                stage="deepstream_runtime",
                message=str(status.get("last_error") or "DeepStream pipeline stopped"),
                status=status,
            )
        if batch is not None:
            return _success_report(status, batch)

    status = dict(backend.status())
    return _failure_report(
        code="DEEPSTREAM_PROBE_TIMEOUT",
        stage="deepstream_runtime",
        message=(
            "DeepStream candidate produced no DetectionBatch within "
            f"{timeout_s:.1f}s; input={int(status.get('input_frames') or 0)} "
            f"output={int(status.get('output_buffers') or 0)} "
            f"parser_calls={int(_mapping(status.get('parser')).get('decode_calls') or 0)}"
        ),
        status=status,
    )


def _success_report(
    status: dict[str, Any],
    batch: Any,
) -> ModelValidationReport:
    parser = _mapping(status.get("parser"))
    inference_stats = _mapping(status.get("nvinfer_total_ms_stats"))
    parser_ok = (
        int(parser.get("parse_failures") or 0) == 0
        and int(status.get("output_buffers") or 0) > 0
    )
    return ModelValidationReport(
        status="validated",
        engine_execution_ok=True,
        output_tensor_ok=parser_ok,
        decoder_ok=parser_ok,
        nms_ok=parser_ok,
        detection_batch_ok=True,
        preprocess_ms=0.0,
        inference_ms=float(inference_stats.get("p50") or 0.0),
        decode_ms=float(parser.get("decode_ms") or 0.0),
        nms_ms=0.0,
        detection_batch=batch,
    )


def _failure_report(
    *,
    code: str,
    stage: str,
    message: str,
    status: dict[str, Any] | None = None,
) -> ModelValidationReport:
    payload = status or {}
    parser = _mapping(payload.get("parser"))
    inference_stats = _mapping(payload.get("nvinfer_total_ms_stats"))
    return ModelValidationReport(
        status="invalid",
        engine_execution_ok=int(payload.get("output_buffers") or 0) > 0,
        output_tensor_ok=False,
        decoder_ok=False,
        nms_ok=False,
        detection_batch_ok=False,
        inference_ms=float(inference_stats.get("p50") or 0.0),
        decode_ms=float(parser.get("decode_ms") or 0.0),
        issues=(ValidationIssue(code=code, stage=stage, message=message),),
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


__all__ = ["probe_deepstream_model"]
