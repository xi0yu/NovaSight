from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from math import isfinite
import time
from typing import Any, Callable, ContextManager

from novasight.contracts import Detection, DetectionBatch

from .profile import (
    ModelProfile,
    ModelStatus,
    ProfileValidation,
    model_profile_validation_fingerprint,
)


@dataclass(frozen=True)
class TensorStatistics:
    name: str
    shape: tuple[int, ...]
    dtype: str
    minimum: float
    maximum: float
    mean: float
    nan_count: int
    inf_count: int


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    stage: str
    message: str


@dataclass(frozen=True)
class ModelValidationReport:
    status: str
    engine_execution_ok: bool
    output_tensor_ok: bool
    decoder_ok: bool
    nms_ok: bool
    detection_batch_ok: bool
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    decode_ms: float = 0.0
    nms_ms: float = 0.0
    tensor_statistics: tuple[TensorStatistics, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()
    detection_batch: DetectionBatch | None = None


class ModelProbe:
    """Validate one candidate engine without making it the active runtime engine."""

    def run(
        self,
        profile: ModelProfile,
        *,
        inference: Any,
        isolation: ContextManager[Any] | None = None,
        frame: Any | None = None,
        frame_supplier: Callable[[], Any] | None = None,
        now_ns: Callable[[], int] = time.monotonic_ns,
    ) -> tuple[ModelProfile, ModelValidationReport]:
        if profile.status not in {ModelStatus.READY_FOR_PROBE, ModelStatus.VALIDATED}:
            raise ValueError(
                "model profile must be READY_FOR_PROBE before diagnostic inference "
                f"(got {profile.status.value})"
            )
        context = isolation if isolation is not None else nullcontext()
        candidate: Any | None = None
        try:
            with context:
                try:
                    prepare_profile = getattr(inference, "prepare_profile", None)
                    if not callable(prepare_profile):
                        raise RuntimeError("inference runtime does not support profile diagnostics")
                    candidate, status = prepare_profile(profile, diagnostic=True)
                    if status.get("loaded") is not True or status.get("warmed") is not True:
                        reason = str(status.get("reason") or "candidate engine did not warm up")
                        raise RuntimeError(reason)
                    if frame is not None:
                        probe_frame = frame
                    elif frame_supplier is not None:
                        probe_frame = frame_supplier()
                        if probe_frame is None:
                            raise RuntimeError("diagnostic input did not provide a frame")
                    else:
                        probe_frame = _zero_frame(profile)
                    inference_start_ns = max(1, int(now_ns()))
                    result = candidate.infer(probe_frame)
                    inference_end_ns = max(inference_start_ns, int(now_ns()))
                    report = _validation_report(
                        profile,
                        result,
                        inference_start_ns=inference_start_ns,
                        inference_end_ns=inference_end_ns,
                    )
                finally:
                    _close_candidate(candidate)
                    candidate = None
        except Exception as exc:
            report = ModelValidationReport(
                status="invalid",
                engine_execution_ok=False,
                output_tensor_ok=False,
                decoder_ok=False,
                nms_ok=False,
                detection_batch_ok=False,
                issues=(
                    ValidationIssue(
                        code="PROBE_FAILED",
                        stage="inference",
                        message=str(exc),
                    ),
                ),
            )
        finally:
            _close_candidate(candidate)

        passed = report.status == "validated"
        next_profile = replace(
            profile,
            status=ModelStatus.VALIDATED if passed else ModelStatus.INVALID,
            validation=ProfileValidation(
                status=report.status,
                validated_at=_iso_timestamp(),
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
        return next_profile, report


def _zero_frame(profile: ModelProfile) -> Any:
    import numpy as np

    _, _, height, width = profile.input.runtime_shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    return type(
        "DiagnosticFrame",
        (),
        {
            "image": image,
            "width": width,
            "height": height,
            "pixel_format": profile.preprocess.color_format,
            "frame_id": 0,
            "capture_ts_ns": 1,
        },
    )()


def _validation_report(
    profile: ModelProfile,
    result: Any,
    *,
    inference_start_ns: int,
    inference_end_ns: int,
) -> ModelValidationReport:
    issues: list[ValidationIssue] = []
    available = bool(getattr(result, "available", False))
    if not available:
        issues.append(
            ValidationIssue(
                code="ENGINE_EXECUTION_FAILED",
                stage="inference",
                message=str(getattr(result, "reason", "") or "candidate inference failed"),
            )
        )
    debug = dict(getattr(result, "debug", {}) or {})
    tensor_statistics = _tensor_statistics(debug.get("tensor_statistics"))
    if not tensor_statistics:
        issues.append(
            ValidationIssue(
                code="TENSOR_STATISTICS_MISSING",
                stage="output",
                message="diagnostic inference did not expose output tensor statistics",
            )
        )
    elif any(stat.nan_count or stat.inf_count for stat in tensor_statistics):
        issues.append(
            ValidationIssue(
                code="NON_FINITE_OUTPUT",
                stage="output",
                message="output tensor contains NaN or Inf values",
            )
        )

    decode = dict(debug.get("decode", {}) or {})
    decode_reason = str(decode.get("reason") or "")
    decoder_ok = available and decode_reason not in {
        "unsupported output dimensions",
        "fewer than 5 prediction columns",
        "no supported decode layout",
    }
    if not decoder_ok:
        issues.append(
            ValidationIssue(
                code="DECODER_FAILED",
                stage="decode",
                message=decode_reason or "decoder did not accept the output contract",
            )
        )
    detections = list(getattr(result, "detections", []) or [])
    detection_issues = _validate_detections(profile, detections)
    issues.extend(detection_issues)
    detection_batch_ok = available and decoder_ok and not detection_issues
    batch = None
    if detection_batch_ok:
        batch = DetectionBatch(
            frame_id=0,
            generation=0,
            capture_ts_ns=inference_start_ns,
            inference_start_ts_ns=inference_start_ns,
            inference_end_ts_ns=inference_end_ns,
            detections=[
                Detection(cls=item.cls, score=item.score, box=item.box)
                for item in detections
            ],
            classes=list(profile.labels),
            coordinate_space="roi",
            publish_ts_ns=inference_end_ns,
            model_input_size=(profile.input.runtime_shape[3], profile.input.runtime_shape[2]),
            roi_size=(profile.input.runtime_shape[3], profile.input.runtime_shape[2]),
            metadata={"diagnostic": True, "control_enabled": False},
        )

    timings = dict(debug.get("timings", {}) or {})
    decode_ms = float(decode.get("decode_ms") or 0.0)
    execute_total_ms = float(timings.get("execute_total_ms") or 0.0)
    output_tensor_ok = bool(tensor_statistics) and not any(
        stat.nan_count or stat.inf_count for stat in tensor_statistics
    )
    nms_ok = decoder_ok and int(decode.get("nms_detections", len(detections)) or 0) >= 0
    passed = available and output_tensor_ok and decoder_ok and nms_ok and detection_batch_ok
    return ModelValidationReport(
        status="validated" if passed else "invalid",
        engine_execution_ok=available,
        output_tensor_ok=output_tensor_ok,
        decoder_ok=decoder_ok,
        nms_ok=nms_ok,
        detection_batch_ok=detection_batch_ok,
        preprocess_ms=(
            float(timings.get("prepare_input_ms") or 0.0)
            + float(timings.get("numpy_tensor_ms") or 0.0)
        ),
        inference_ms=max(0.0, execute_total_ms - decode_ms),
        decode_ms=decode_ms,
        nms_ms=float(decode.get("nms_ms") or 0.0),
        tensor_statistics=tensor_statistics,
        issues=tuple(issues),
        detection_batch=batch,
    )


def _tensor_statistics(raw: Any) -> tuple[TensorStatistics, ...]:
    if not isinstance(raw, list):
        return ()
    statistics: list[TensorStatistics] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        statistics.append(
            TensorStatistics(
                name=str(item.get("name") or ""),
                shape=tuple(int(value) for value in item.get("shape", [])),
                dtype=str(item.get("dtype") or ""),
                minimum=float(item.get("minimum") or 0.0),
                maximum=float(item.get("maximum") or 0.0),
                mean=float(item.get("mean") or 0.0),
                nan_count=int(item.get("nan_count") or 0),
                inf_count=int(item.get("inf_count") or 0),
            )
        )
    return tuple(statistics)


def _validate_detections(profile: ModelProfile, detections: list[Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    _, _, height, width = profile.input.runtime_shape
    for index, detection in enumerate(detections):
        values = (
            float(detection.score),
            float(detection.box.x1),
            float(detection.box.y1),
            float(detection.box.x2),
            float(detection.box.y2),
        )
        if not all(isfinite(value) for value in values):
            issues.append(
                ValidationIssue("NON_FINITE_DETECTION", "decode", f"detection {index} is non-finite")
            )
            continue
        if not 0.0 <= float(detection.score) <= 1.0:
            issues.append(
                ValidationIssue("INVALID_CONFIDENCE", "decode", f"detection {index} score is invalid")
            )
        if int(detection.cls) < 0 or int(detection.cls) >= profile.decoder.class_count:
            issues.append(
                ValidationIssue("CLASS_ID_OUT_OF_RANGE", "decode", f"detection {index} class is invalid")
            )
        box = detection.box
        if (
            box.x1 < 0
            or box.y1 < 0
            or box.x2 > width
            or box.y2 > height
            or box.x2 <= box.x1
            or box.y2 <= box.y1
        ):
            issues.append(
                ValidationIssue("BBOX_OUT_OF_RANGE", "decode", f"detection {index} bbox is invalid")
            )
    return issues


def _iso_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _close_candidate(candidate: Any | None) -> None:
    close = getattr(candidate, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


__all__ = [
    "ModelProbe",
    "ModelValidationReport",
    "TensorStatistics",
    "ValidationIssue",
]
