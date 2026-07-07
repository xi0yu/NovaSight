"""Shared inference postprocessing utilities."""

from .yolo import decode_nx6_detections

__all__ = ["decode_nx6_detections"]
