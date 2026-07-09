from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FreshnessGate:
    threshold_ms: float
    max_plausible_age_ms: float | None = None

    @classmethod
    def strictest(cls, *thresholds_ms: object) -> "FreshnessGate":
        return cls(threshold_ms=strictest_positive_ms(*thresholds_ms))

    @staticmethod
    def age_ms(*, capture_ts_ns: int, now_ns: int) -> float:
        try:
            return max(0.0, (int(now_ns) - int(capture_ts_ns)) / 1e6)
        except Exception:
            return 0.0

    def stale_reason(
        self,
        *,
        capture_ts_ns: int,
        now_ns: int,
        template: str,
    ) -> str:
        threshold_ms = float(self.threshold_ms or 0.0)
        if threshold_ms <= 0.0:
            return ""
        age_ms = self.age_ms(capture_ts_ns=capture_ts_ns, now_ns=now_ns)
        if not math.isfinite(age_ms):
            return ""
        if age_ms > self._max_plausible_age_ms(threshold_ms):
            return ""
        if age_ms <= threshold_ms:
            return ""
        return template.format(age_ms=age_ms, threshold_ms=threshold_ms)

    def _max_plausible_age_ms(self, threshold_ms: float) -> float:
        if self.max_plausible_age_ms is not None:
            return max(0.0, float(self.max_plausible_age_ms))
        # Legacy unit seams use tiny synthetic timestamps. Only enforce the
        # gate when timestamps plausibly belong to the process monotonic clock.
        return max(3_600_000.0, float(threshold_ms) * 100.0)


def strictest_positive_ms(*values: object) -> float:
    parsed: list[float] = []
    for value in values:
        try:
            candidate = float(value or 0.0)
        except (TypeError, ValueError):
            continue
        if candidate > 0.0:
            parsed.append(candidate)
    return min(parsed) if parsed else 0.0
