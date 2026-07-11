from __future__ import annotations

from typing import Any

from novasight.deepstream.runtime_pipeline import create_deepstream_runtime_pipeline

from .pipeline import RuntimePipeline


def create_runtime_pipeline(*, capture: Any, runtime: Any) -> Any:
    backend = str(getattr(runtime.config.inference, "backend", "")).lower()
    if backend == "deepstream_nvinfer":
        return create_deepstream_runtime_pipeline(runtime=runtime)
    return RuntimePipeline(capture=capture, runtime=runtime)


__all__ = ["create_runtime_pipeline"]
