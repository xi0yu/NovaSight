from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from io import BytesIO
import colorsys
import json
import math
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from novasight.config import CrosshairConfig


@dataclass(frozen=True, slots=True)
class CrosshairObservation:
    status: str
    found: bool
    x: float
    y: float
    confidence: float
    sample_ts_ns: int
    template_id: str
    point_hits: int
    point_count: int
    offset_x: float
    offset_y: float
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ControlReference:
    x: float
    y: float
    source: str
    confidence: float
    sample_ts_ns: int
    age_ms: float
    geometry_signature: str
    reason: str = ""


class CrosshairSystem:
    """Learn, verify, and resolve one stable control reference.

    The external interface intentionally exposes observations and one resolved
    reference. HSV extraction, template points, confirmation timing, persistence,
    and fallback policy stay inside this module.
    """

    def __init__(
        self,
        config: CrosshairConfig,
        *,
        template_path: Path | str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._mutation_lock = threading.Lock()
        self.config = config
        self.template_path = Path(template_path) if template_path is not None else None
        self._recent: deque[tuple[int, int, int, Image.Image]] = deque(
            maxlen=max(3, int(config.sample_frames))
        )
        self._template: dict[str, Any] | None = None
        self._latest: CrosshairObservation | None = None
        self._last_confirmed: CrosshairObservation | None = None
        self._candidate_xy: tuple[float, float] | None = None
        self._candidate_since_ns = 0
        self._processed_frames = 0
        self._matched_frames = 0
        self._last_error = ""
        self._load_template()

    def update_config(self, config: CrosshairConfig) -> None:
        with self._mutation_lock:
            with self._lock:
                search_changed = int(config.search_size) != int(self.config.search_size)
                sample_frames_changed = int(config.sample_frames) != int(
                    self.config.sample_frames
                )
                self.config = config
                if sample_frames_changed:
                    items = list(self._recent)[-max(3, int(config.sample_frames)) :]
                    self._recent = deque(items, maxlen=max(3, int(config.sample_frames)))
                if search_changed:
                    self._clear_runtime_locked()

    def ingest_jpeg(
        self,
        payload: bytes,
        *,
        sample_ts_ns: int,
        roi_width: int,
        roi_height: int,
    ) -> CrosshairObservation:
        timestamp = int(sample_ts_ns)
        if timestamp <= 0:
            raise ValueError("crosshair sample_ts_ns must be positive")
        with self._lock:
            config = self.config
        image = _decode_search_image(payload, int(config.search_size))
        with self._lock:
            if self.config is not config:
                return self._store_observation_locked(
                    _empty_observation("searching", timestamp, "config_changed")
                )
            self._processed_frames += 1
            self._last_error = ""
            self._recent.append((timestamp, int(roi_width), int(roi_height), image))
            template = self._template
            if not config.enabled:
                return self._store_observation_locked(
                    _empty_observation("disabled", timestamp, "CROSSHAIR_DISABLED")
                )
            if template is None:
                return self._store_observation_locked(
                    _empty_observation("searching", timestamp, "template_unavailable")
                )
            geometry_signature = _geometry_signature(roi_width, roi_height)
            if str(template.get("geometry_signature")) != geometry_signature:
                return self._store_observation_locked(
                    _empty_observation("searching", timestamp, "geometry_changed")
                )
            if int(template.get("search_size", 0)) != image.width:
                return self._store_observation_locked(
                    _empty_observation("searching", timestamp, "search_size_changed")
                )
            hint_offset = (
                (
                    round(self._last_confirmed.offset_x),
                    round(self._last_confirmed.offset_y),
                )
                if self._latest is not None
                and self._latest.status == "confirmed"
                and self._last_confirmed is not None
                else None
            )

        # JPEG decoding and matching never hold the state lock used by the
        # latency-sensitive resolve() call.
        match = _match_template(
            image,
            template,
            max_offset=int(round(float(config.max_offset_px))),
            hint_offset=hint_offset,
        )
        with self._lock:
            if self.config is not config or self._template is not template:
                return self._store_observation_locked(
                    _empty_observation("searching", timestamp, "template_changed")
                )
            if (
                match is None
                or match[2] < float(config.min_similarity)
                or match[4] < 0.015
            ):
                self._candidate_xy = None
                self._candidate_since_ns = 0
                status = "uncertain" if self._last_confirmed is not None else "searching"
                return self._store_observation_locked(
                    _empty_observation(status, timestamp, "template_not_matched")
                )

            offset_x, offset_y, confidence, hits, _uniqueness = match
            x = (int(roi_width) - image.width) * 0.5 + image.width * 0.5 + offset_x
            y = (int(roi_height) - image.height) * 0.5 + image.height * 0.5 + offset_y
            current_xy = (x, y)
            if self._candidate_xy is None or _distance(current_xy, self._candidate_xy) > 2.0:
                self._candidate_xy = current_xy
                self._candidate_since_ns = timestamp
            confirmed = (
                timestamp - self._candidate_since_ns
                >= int(max(0.0, float(config.confirm_duration_ms)) * 1_000_000.0)
            )
            status = "confirmed" if confirmed else "candidate"
            if confirmed and self._last_confirmed is not None:
                x, y = _limited_point(
                    previous=(self._last_confirmed.x, self._last_confirmed.y),
                    requested=(x, y),
                    max_step=float(config.max_step_px),
                )
            observation = CrosshairObservation(
                status=status,
                found=True,
                x=x,
                y=y,
                confidence=confidence,
                sample_ts_ns=timestamp,
                template_id=str(template["id"]),
                point_hits=hits,
                point_count=len(template["points"]),
                offset_x=x - int(roi_width) * 0.5,
                offset_y=y - int(roi_height) * 0.5,
            )
            self._matched_frames += 1
            if confirmed:
                self._last_confirmed = observation
            return self._store_observation_locked(observation)

    def learn(self) -> dict[str, Any]:
        with self._mutation_lock:
            with self._lock:
                config = self.config
                required = max(3, int(config.sample_frames))
                if len(self._recent) < required:
                    raise ValueError(
                        f"crosshair learning requires {required} recent samples; "
                        f"only {len(self._recent)} available"
                    )
                frames = list(self._recent)[-required:]
            roi_width = frames[-1][1]
            roi_height = frames[-1][2]
            if any(width != roi_width or height != roi_height for _, width, height, _ in frames):
                raise ValueError("crosshair learning samples changed ROI geometry")
            now_ns = time.monotonic_ns()
            newest_age_ms = (now_ns - frames[-1][0]) / 1_000_000.0
            if newest_age_ms < 0.0 or newest_age_ms > float(config.max_age_ms):
                raise ValueError("crosshair learning samples are stale; wait for live frames")
            maximum_span_ms = max(
                float(config.max_age_ms),
                (required - 1) * 3_000.0 / max(1, int(config.sample_hz)),
            )
            if (frames[-1][0] - frames[0][0]) / 1_000_000.0 > maximum_span_ms:
                raise ValueError("crosshair learning samples span too much time")
            median = _median_image([item[3] for item in frames])
            points = _learn_template_points(median)
            if len(points) < 8:
                raise ValueError(
                    "crosshair learning could not isolate enough stable center structure"
                )
            template = {
                "schema_version": 1,
                "id": uuid4().hex[:12],
                "created_ts_ns": time.time_ns(),
                "geometry_signature": _geometry_signature(roi_width, roi_height),
                "search_size": median.width,
                "points": points,
            }
            validation_match = _match_template(
                median,
                template,
                max_offset=min(20, max(3, int(round(float(config.max_offset_px))))),
            )
            if (
                validation_match is None
                or abs(validation_match[0]) > 1
                or abs(validation_match[1]) > 1
                or validation_match[4] < 0.015
            ):
                raise ValueError(
                    "crosshair learning structure is not unique near screen center"
                )
            self._save_template(template)
            with self._lock:
                if self.config is not config:
                    raise ValueError("crosshair config changed during learning; try again")
                self._template = template
                self._clear_tracking_locked()
            return _template_summary(template)

    def clear_template(self) -> dict[str, Any]:
        with self._mutation_lock:
            with self._lock:
                self._template = None
                self._clear_runtime_locked()
            if self.template_path is not None:
                self.template_path.unlink(missing_ok=True)
            return self.status()

    def resolve(
        self,
        *,
        geometric_x: float,
        geometric_y: float,
        now_ns: int,
        geometry_signature: str,
    ) -> ControlReference:
        with self._lock:
            fallback_reason = self._fallback_reason_locked(
                now_ns=int(now_ns),
                geometry_signature=str(geometry_signature),
            )
            if fallback_reason:
                return ControlReference(
                    x=float(geometric_x),
                    y=float(geometric_y),
                    source="geometry",
                    confidence=1.0,
                    sample_ts_ns=0,
                    age_ms=0.0,
                    geometry_signature=str(geometry_signature),
                    reason=fallback_reason,
                )
            assert self._last_confirmed is not None
            age_ms = max(
                0.0,
                (int(now_ns) - self._last_confirmed.sample_ts_ns) / 1_000_000.0,
            )
            source = (
                "vision_verified"
                if self._latest is not None and self._latest.status == "confirmed"
                else "vision_hold"
            )
            return ControlReference(
                x=self._last_confirmed.x,
                y=self._last_confirmed.y,
                source=source,
                confidence=self._last_confirmed.confidence,
                sample_ts_ns=self._last_confirmed.sample_ts_ns,
                age_ms=age_ms,
                geometry_signature=str(geometry_signature),
            )

    def reset_runtime(self) -> None:
        with self._lock:
            self._clear_runtime_locked()

    def record_error(self, message: str) -> None:
        with self._lock:
            self._last_error = str(message).strip()

    def status(self) -> dict[str, Any]:
        with self._lock:
            geometry_signature = (
                _geometry_signature(self._recent[-1][1], self._recent[-1][2])
                if self._recent
                else str((self._template or {}).get("geometry_signature", ""))
            )
            reference_reason = self._fallback_reason_locked(
                now_ns=time.monotonic_ns(),
                geometry_signature=geometry_signature,
            )
            return {
                "enabled": bool(self.config.enabled),
                "use_for_control": bool(self.config.use_for_control),
                "state": self._latest.status if self._latest is not None else "idle",
                "observation": asdict(self._latest) if self._latest is not None else None,
                "control_reference_ready": not reference_reason,
                "control_reference_source": (
                    "geometry"
                    if reference_reason
                    else (
                        "vision_verified"
                        if self._latest is not None and self._latest.status == "confirmed"
                        else "vision_hold"
                    )
                ),
                "control_reference_reason": reference_reason,
                "recent_samples": len(self._recent),
                "required_samples": max(3, int(self.config.sample_frames)),
                "processed_frames": self._processed_frames,
                "matched_frames": self._matched_frames,
                "last_error": self._last_error,
                "template": (
                    _template_summary(self._template) if self._template is not None else None
                ),
            }

    def template_preview_png(self) -> bytes | None:
        with self._lock:
            if self._template is None:
                return None
            template = self._template
        size = int(template["search_size"])
        image = Image.new("RGBA", (size, size), (15, 18, 25, 255))
        draw = ImageDraw.Draw(image)
        center = size // 2
        for point in template["points"]:
            x = center + int(point[0])
            y = center + int(point[1])
            draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=(255, 67, 113, 255))
        draw.line((center - 4, center, center + 4, center), fill=(255, 255, 255, 180))
        draw.line((center, center - 4, center, center + 4), fill=(255, 255, 255, 180))
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def _fallback_reason_locked(self, *, now_ns: int, geometry_signature: str) -> str:
        if not self.config.enabled:
            return "disabled"
        if not self.config.use_for_control:
            return "control_disabled"
        if self._template is None:
            return "template_unavailable"
        if str(self._template.get("geometry_signature")) != geometry_signature:
            return "geometry_changed"
        if int(self._template.get("search_size", 0)) != int(self.config.search_size):
            return "search_size_changed"
        if self._last_confirmed is None:
            return "observation_unconfirmed"
        age_ms = (now_ns - self._last_confirmed.sample_ts_ns) / 1_000_000.0
        if age_ms < 0.0 or age_ms > float(self.config.max_age_ms):
            return "observation_stale"
        return ""

    def _store_observation_locked(
        self,
        observation: CrosshairObservation,
    ) -> CrosshairObservation:
        self._latest = observation
        return observation

    def _clear_tracking_locked(self) -> None:
        self._latest = None
        self._last_confirmed = None
        self._candidate_xy = None
        self._candidate_since_ns = 0

    def _clear_runtime_locked(self) -> None:
        self._recent.clear()
        self._clear_tracking_locked()
        self._last_error = ""

    def _load_template(self) -> None:
        if self.template_path is None or not self.template_path.is_file():
            return
        try:
            payload = json.loads(self.template_path.read_text(encoding="utf-8"))
            _validate_template(payload)
            self._template = payload
        except Exception as exc:
            self._last_error = f"template load failed: {exc}"

    def _save_template(self, template: dict[str, Any]) -> None:
        if self.template_path is None:
            return
        _validate_template(template)
        self.template_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.template_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(template, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(self.template_path)


def _validate_template(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("crosshair template must be a mapping")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported crosshair template schema")
    template_id = payload.get("id")
    if not isinstance(template_id, str) or not template_id or len(template_id) > 64:
        raise ValueError("crosshair template id is invalid")
    search_size = payload.get("search_size")
    if not isinstance(search_size, int) or search_size < 32 or search_size > 640:
        raise ValueError("crosshair template search_size is invalid")
    geometry = payload.get("geometry_signature")
    if not isinstance(geometry, str) or "x" not in geometry:
        raise ValueError("crosshair template geometry is invalid")
    try:
        geometry_width, geometry_height = (int(value) for value in geometry.split("x", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("crosshair template geometry is invalid") from exc
    if geometry_width <= 0 or geometry_height <= 0:
        raise ValueError("crosshair template geometry is invalid")
    points = payload.get("points")
    if not isinstance(points, list) or not 8 <= len(points) <= 256:
        raise ValueError("crosshair template points are invalid")
    radius = search_size // 2
    for point in points:
        if (
            not isinstance(point, list)
            or len(point) != 6
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in point)
        ):
            raise ValueError("crosshair template point format is invalid")
        dx, dy, hue, saturation, value, gray = point
        if abs(dx) >= radius or abs(dy) >= radius:
            raise ValueError("crosshair template point lies outside search area")
        if not all(0 <= channel <= 255 for channel in (hue, saturation, value, gray)):
            raise ValueError("crosshair template point channel is invalid")


def _decode_search_image(payload: bytes, search_size: int) -> Image.Image:
    if not payload:
        raise ValueError("crosshair JPEG payload is empty")
    try:
        with Image.open(BytesIO(payload)) as source:
            image = source.convert("RGB")
    except Exception as exc:
        raise ValueError(f"crosshair JPEG decode failed: {exc}") from exc
    if image.width != search_size or image.height != search_size:
        side = min(image.width, image.height)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side)).resize(
            (search_size, search_size),
            Image.Resampling.BILINEAR,
        )
    return image


def _median_image(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("crosshair median requires images")
    size = images[0].size
    if any(image.size != size for image in images):
        raise ValueError("crosshair learning images must share one size")
    stack = np.stack([np.asarray(image, dtype=np.uint8) for image in images], axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)
    return Image.fromarray(median, mode="RGB")


def _learn_template_points(image: Image.Image) -> list[list[int]]:
    width, height = image.size
    cx, cy = width // 2, height // 2
    radius = min(24, width // 3, height // 3)
    pixels = image.load()
    candidates: dict[tuple[int, int], tuple[float, list[int]]] = {}
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x, y = cx + dx, cy + dy
            r, g, b = pixels[x, y]
            h, s, v = _rgb_to_hsv255(r, g, b)
            gray = _gray(r, g, b)
            neighbors = [
                _gray(*pixels[min(width - 1, max(0, x + ox)), min(height - 1, max(0, y + oy))])
                for ox, oy in ((-3, 0), (3, 0), (0, -3), (0, 3))
            ]
            contrast = abs(gray - sorted(neighbors)[len(neighbors) // 2])
            if contrast < 18 and not (s >= 70 and v >= 70):
                continue
            centrality = max(0.0, 1.0 - math.hypot(dx, dy) / max(1.0, radius))
            score = contrast + s * 0.35 + centrality * 12.0
            candidates[(dx, dy)] = (score, [dx, dy, h, s, v, gray])

    symmetric: list[tuple[float, list[int]]] = []
    for (dx, dy), value in candidates.items():
        if any(
            (-dx + ox, -dy + oy) in candidates
            for ox in (-1, 0, 1)
            for oy in (-1, 0, 1)
        ):
            symmetric.append(value)
    if len(symmetric) < 8:
        return []
    selected = symmetric
    selected.sort(key=lambda item: item[0], reverse=True)
    # A sparse signature is enough for a HUD element and keeps the observer
    # comfortably outside the latency-sensitive inference/control path.
    return [point for _score, point in selected[:96]]


def _match_template(
    image: Image.Image,
    template: dict[str, Any],
    *,
    max_offset: int,
    hint_offset: tuple[int, int] | None = None,
) -> tuple[int, int, float, int, float] | None:
    points = template.get("points")
    if not isinstance(points, list) or not points:
        return None
    point_array = np.asarray(points, dtype=np.int16)
    hsv_image = np.asarray(image.convert("HSV"), dtype=np.int16)
    gray_image = np.asarray(image.convert("L"), dtype=np.int16)
    cx, cy = image.width // 2, image.height // 2

    def score_offsets(offsets: list[tuple[int, int]]) -> list[tuple[int, int, float, int]]:
        if not offsets:
            return []
        offset_array = np.asarray(offsets, dtype=np.int16)
        xs = cx + point_array[None, :, 0] + offset_array[:, None, 0]
        ys = cy + point_array[None, :, 1] + offset_array[:, None, 1]
        valid = (xs >= 0) & (xs < image.width) & (ys >= 0) & (ys < image.height)
        safe_xs = np.clip(xs, 0, image.width - 1)
        safe_ys = np.clip(ys, 0, image.height - 1)
        sampled_hsv = hsv_image[safe_ys, safe_xs]
        sampled_gray = gray_image[safe_ys, safe_xs]
        template_hsv = point_array[None, :, 2:5]
        template_gray = point_array[None, :, 5]
        hue_difference = np.abs(sampled_hsv[:, :, 0] - template_hsv[:, :, 0])
        hue_distance = np.minimum(hue_difference, 256 - hue_difference)
        gray_score = np.maximum(
            0.0,
            1.0 - np.abs(sampled_gray - template_gray) / 80.0,
        )
        color_score = (
            np.maximum(0.0, 1.0 - hue_distance / 32.0) * 0.45
            + np.maximum(
                0.0,
                1.0 - np.abs(sampled_hsv[:, :, 1] - template_hsv[:, :, 1]) / 110.0,
            )
            * 0.20
            + np.maximum(
                0.0,
                1.0 - np.abs(sampled_hsv[:, :, 2] - template_hsv[:, :, 2]) / 100.0,
            )
            * 0.15
            + gray_score * 0.20
        )
        scores = np.where(template_hsv[:, :, 1] >= 50, color_score, gray_score)
        scores = np.where(valid, scores, 0.0)
        compared = valid.sum(axis=1)
        confidence = scores.sum(axis=1) / np.maximum(1, compared)
        hits = ((scores >= 0.60) & valid).sum(axis=1)
        minimum_compared = max(4, len(points) // 2)
        return [
            (int(offset_x), int(offset_y), float(confidence[index]), int(hits[index]))
            for index, (offset_x, offset_y) in enumerate(offsets)
            if int(compared[index]) >= minimum_compared
        ]

    def finalize(
        candidates: list[tuple[int, int, float, int]],
    ) -> tuple[int, int, float, int, float] | None:
        if not candidates:
            return None
        best = max(
            candidates,
            key=lambda item: item[2]
            - 0.002 * math.hypot(item[0], item[1]) / max(1.0, float(max_offset)),
        )
        competitors = [
            candidate
            for candidate in candidates
            if math.hypot(candidate[0] - best[0], candidate[1] - best[1]) >= 3.0
        ]
        competing_score = max((candidate[2] for candidate in competitors), default=0.0)
        return best[0], best[1], best[2], best[3], max(0.0, best[2] - competing_score)

    if hint_offset is not None:
        hint_x = max(-max_offset, min(max_offset, int(hint_offset[0])))
        hint_y = max(-max_offset, min(max_offset, int(hint_offset[1])))
        offsets = [
            (offset_x, offset_y)
            for offset_y in range(
                max(-max_offset, hint_y - 3),
                min(max_offset, hint_y + 3) + 1,
            )
            for offset_x in range(
                max(-max_offset, hint_x - 3),
                min(max_offset, hint_x + 3) + 1,
            )
        ]
        candidates = score_offsets(offsets)
        return finalize(candidates)

    # Crosshairs stay close to screen centre. Search the full configured range
    # on a coarse grid, then refine only around its strongest result. This
    # preserves the recovery range without paying for every integer offset.
    coarse_step = 4 if max_offset >= 8 else 2
    coarse_offsets = list(range(-max_offset, max_offset + 1, coarse_step))
    if coarse_offsets[-1] != max_offset:
        coarse_offsets.append(max_offset)
    candidates = score_offsets(
        [(offset_x, offset_y) for offset_y in coarse_offsets for offset_x in coarse_offsets]
    )
    best = finalize(candidates)
    if best is None:
        return None

    coarse_x, coarse_y = best[0], best[1]
    fine_offsets = [
        (offset_x, offset_y)
        for offset_y in range(
            max(-max_offset, coarse_y - coarse_step),
            min(max_offset, coarse_y + coarse_step) + 1,
        )
        for offset_x in range(
            max(-max_offset, coarse_x - coarse_step),
            min(max_offset, coarse_x + coarse_step) + 1,
        )
    ]
    candidates.extend(score_offsets(fine_offsets))
    return finalize(candidates)


def _template_summary(template: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(template["id"]),
        "schema_version": int(template["schema_version"]),
        "created_ts_ns": int(template["created_ts_ns"]),
        "geometry_signature": str(template["geometry_signature"]),
        "search_size": int(template["search_size"]),
        "point_count": len(template["points"]),
    }


def _empty_observation(status: str, sample_ts_ns: int, reason: str) -> CrosshairObservation:
    return CrosshairObservation(
        status=status,
        found=False,
        x=0.0,
        y=0.0,
        confidence=0.0,
        sample_ts_ns=int(sample_ts_ns),
        template_id="",
        point_hits=0,
        point_count=0,
        offset_x=0.0,
        offset_y=0.0,
        reason=reason,
    )


def _rgb_to_hsv255(r: int, g: int, b: int) -> tuple[int, int, int]:
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return round(h * 255.0) % 256, round(s * 255.0), round(v * 255.0)


def _gray(r: int, g: int, b: int) -> int:
    return round(r * 0.299 + g * 0.587 + b * 0.114)


def _geometry_signature(width: int, height: int) -> str:
    return f"{int(width)}x{int(height)}"


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _limited_point(
    *,
    previous: tuple[float, float],
    requested: tuple[float, float],
    max_step: float,
) -> tuple[float, float]:
    distance = _distance(previous, requested)
    if distance <= max_step or distance <= 0.0:
        return requested
    scale = max_step / distance
    return (
        previous[0] + (requested[0] - previous[0]) * scale,
        previous[1] + (requested[1] - previous[1]) * scale,
    )
