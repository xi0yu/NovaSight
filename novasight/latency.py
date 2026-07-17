from __future__ import annotations

from typing import TypedDict


class PipelineLatencyBreakdown(TypedDict):
    ingress_ms: float
    inference_ms: float
    batch_build_ms: float
    publish_age_ms: float
    control_wait_ms: float
    control_ms: float
    accounted_ms: float
    total_ms: float
    unattributed_ms: float


def pipeline_latency_breakdown_ms(
    *,
    capture_ts_ns: int,
    inference_start_ts_ns: int,
    inference_end_ts_ns: int,
    publish_ts_ns: int,
    control_start_ts_ns: int | None = None,
    control_end_ts_ns: int | None = None,
) -> PipelineLatencyBreakdown:
    """Return one non-overlapping capture-to-control latency timeline.

    Every stage is derived from adjacent timestamps in the same monotonic clock
    domain. Invalid ordering is rejected instead of being hidden by clamping.
    """

    if (control_start_ts_ns is None) != (control_end_ts_ns is None):
        raise ValueError("control start and end timestamps must be provided together")
    control_start = int(publish_ts_ns if control_start_ts_ns is None else control_start_ts_ns)
    control_end = int(control_start if control_end_ts_ns is None else control_end_ts_ns)
    ordered = (
        ("capture", int(capture_ts_ns)),
        ("inference_start", int(inference_start_ts_ns)),
        ("inference_end", int(inference_end_ts_ns)),
        ("publish", int(publish_ts_ns)),
        ("control_start", control_start),
        ("control_end", control_end),
    )
    for (left_name, left), (right_name, right) in zip(ordered, ordered[1:]):
        if left <= 0 or right < left:
            raise ValueError(
                "pipeline timestamps must be positive and monotonic: "
                f"{left_name}={left} {right_name}={right}"
            )

    def elapsed_ms(start: int, end: int) -> float:
        return (end - start) / 1e6

    ingress_ms = elapsed_ms(ordered[0][1], ordered[1][1])
    inference_ms = elapsed_ms(ordered[1][1], ordered[2][1])
    batch_build_ms = elapsed_ms(ordered[2][1], ordered[3][1])
    control_wait_ms = elapsed_ms(ordered[3][1], ordered[4][1])
    control_ms = elapsed_ms(ordered[4][1], ordered[5][1])
    publish_age_ms = elapsed_ms(ordered[0][1], ordered[3][1])
    total_ms = elapsed_ms(ordered[0][1], ordered[5][1])
    accounted_ms = (
        ingress_ms
        + inference_ms
        + batch_build_ms
        + control_wait_ms
        + control_ms
    )
    unattributed_ms = total_ms - accounted_ms
    if abs(unattributed_ms) < 1e-9:
        unattributed_ms = 0.0
    return {
        "ingress_ms": ingress_ms,
        "inference_ms": inference_ms,
        "batch_build_ms": batch_build_ms,
        "publish_age_ms": publish_age_ms,
        "control_wait_ms": control_wait_ms,
        "control_ms": control_ms,
        "accounted_ms": accounted_ms,
        "total_ms": total_ms,
        "unattributed_ms": unattributed_ms,
    }
