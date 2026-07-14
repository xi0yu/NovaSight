from __future__ import annotations

import copy
import threading

from novasight.config import RuntimeConfig
from novasight.control.registry import (
    CALIBRATED_ANGULAR,
    DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2,
)


class RuntimeConfigStore:
    def __init__(self, config: RuntimeConfig) -> None:
        self._config = copy.deepcopy(config)
        self._lock = threading.RLock()
        self.version = 0

    def snapshot(self) -> RuntimeConfig:
        with self._lock:
            return copy.deepcopy(self._config)

    def replace(self, config: RuntimeConfig) -> RuntimeConfig:
        with self._lock:
            self._config = copy.deepcopy(config)
            self.version += 1
            return copy.deepcopy(self._config)

    def status(self) -> dict:
        with self._lock:
            config = self._config
            algorithm_id = str(config.control.active_algorithm)
            if algorithm_id == DUAL_PHASE_ATAN_ROBUST_PREDICTIVE_V2:
                projection = config.control.dual_phase_atan_robust_predictive_v2.projection
                fov_x_deg = projection.fov_x_deg
                counts_per_360_x = projection.counts_per_360
                counts_per_360_y = projection.counts_per_360
                invert_y = projection.invert_y
            elif algorithm_id == CALIBRATED_ANGULAR:
                calibrated = config.control.calibrated_angular
                fov_x_deg = calibrated.fov_x_deg
                counts_per_360_x = calibrated.counts_per_360_x
                counts_per_360_y = calibrated.counts_per_360_y
                invert_y = config.control.shared.invert_y
            else:
                fov_x_deg = None
                counts_per_360_x = None
                counts_per_360_y = None
                invert_y = config.control.shared.invert_y
            return {
                "version": self.version,
                "source": {
                    "default": config.source.default,
                    "image_path": config.source.image_path,
                    "image_fps": config.source.image_fps,
                },
                "capture": {
                    "device": config.capture.device,
                    "backend": config.capture.backend,
                    "memory": config.capture.memory,
                    "latest_only": config.capture.latest_only,
                    "appsink_max_buffers": config.capture.appsink_max_buffers,
                    "queue_leaky": config.capture.queue_leaky,
                    "pixel_format": config.capture.pixel_format,
                    "width": config.capture.width,
                    "height": config.capture.height,
                    "fps": config.capture.fps,
                },
                "preprocess": {
                    "backend": config.preprocess.backend,
                    "input_format": config.preprocess.input_format,
                    "output_dtype": config.preprocess.output_dtype,
                    "normalize": config.preprocess.normalize,
                    "use_pinned_memory": config.preprocess.use_pinned_memory,
                    "h2d_async": config.preprocess.h2d_async,
                },
                "runtime": {
                    "freshness_threshold_ms": config.runtime.freshness_threshold_ms,
                    "drop_stale_batches": config.runtime.drop_stale_batches,
                    "consume_latest_only": config.runtime.consume_latest_only,
                },
                "roi": {
                    "size": config.roi.size,
                },
                "roi_size": config.roi.size,
                "calibration": {
                    "profile_id": config.calibration.profile_id,
                    "profile_version": config.calibration.profile_version,
                    "game_sensitivity_fingerprint": config.calibration.game_sensitivity_fingerprint,
                    "control_mode": config.control.active_algorithm,
                    "fov_x_deg": fov_x_deg,
                    "counts_per_360_x": counts_per_360_x,
                    "counts_per_360_y": counts_per_360_y,
                    "invert_y": invert_y,
                },
                "consumers": {
                    "preview": config.consumers.preview,
                    "inference": config.consumers.inference,
                    "recording": config.consumers.recording,
                    "recording_format": config.consumers.recording_format,
                    "recording_path": config.consumers.recording_path,
                },
            }
