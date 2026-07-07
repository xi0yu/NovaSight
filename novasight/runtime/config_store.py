from __future__ import annotations

import copy
import threading

from novasight.config import RuntimeConfig


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
            return {
                "version": self.version,
                "source": {
                    "default": config.source.default,
                    "image_path": config.source.image_path,
                    "image_fps": config.source.image_fps,
                },
                "capture": {
                    "device": config.capture.device,
                    "memory": config.capture.memory,
                    "pixel_format": config.capture.pixel_format,
                    "width": config.capture.width,
                    "height": config.capture.height,
                    "fps": config.capture.fps,
                },
                "roi": {
                    "size": config.roi.size,
                    "offset_x": config.roi.offset_x,
                    "offset_y": config.roi.offset_y,
                },
                "roi_size": config.roi.size,
                "calibration": {
                    "profile_id": config.calibration.profile_id,
                    "profile_version": config.calibration.profile_version,
                    "fov_x_deg": config.calibration.fov_x_deg,
                    "counts_per_360_x": config.calibration.counts_per_360_x,
                    "counts_per_360_y": config.calibration.counts_per_360_y,
                    "axis_sign_x": config.calibration.axis_sign_x,
                    "axis_sign_y": config.calibration.axis_sign_y,
                    "game_sensitivity_fingerprint": config.calibration.game_sensitivity_fingerprint,
                    "projection_profile": config.calibration.projection_profile,
                },
                "consumers": {
                    "preview": config.consumers.preview,
                    "inference": config.consumers.inference,
                    "recording": config.consumers.recording,
                    "recording_format": config.consumers.recording_format,
                    "recording_path": config.consumers.recording_path,
                },
            }
