from __future__ import annotations

from pathlib import Path

from .fingerprint import sha256_text, stable_json
from .manifest import ModelManifest


DEEPSTREAM_CONFIG_TEMPLATE_VERSION = "nvinfer-template-v1"


def generate_nvinfer_config(
    manifest: ModelManifest,
    *,
    engine_path: Path,
) -> tuple[str, str]:
    config = _config_payload(manifest, engine_path=Path(engine_path).resolve())
    text = _render_config(config)
    fingerprint_payload = {
        "template": DEEPSTREAM_CONFIG_TEMPLATE_VERSION,
        "model_fingerprint": manifest.model_fingerprint,
        "input": manifest.input.__dict__,
        "output": manifest.output.__dict__,
        "deepstream": manifest.deepstream.__dict__,
    }
    return text, f"{DEEPSTREAM_CONFIG_TEMPLATE_VERSION}:{sha256_text(stable_json(fingerprint_payload))}"


def _config_payload(manifest: ModelManifest, *, engine_path: Path) -> dict[str, str]:
    return {
        "gpu-id": "0",
        "model-engine-file": str(engine_path),
        "batch-size": str(manifest.runtime.batch_size),
        "network-mode": _network_mode(manifest.runtime.precision),
        "network-type": str(manifest.deepstream.network_type),
        "process-mode": "1",
        "gie-unique-id": str(manifest.deepstream.gie_unique_id),
        "interval": str(manifest.deepstream.interval),
        "net-scale-factor": _format_float(manifest.input.scale_factor),
        "model-color-format": _model_color_format(manifest.input.color_format),
        "maintain-aspect-ratio": _bool_int(manifest.input.maintain_aspect_ratio),
        "symmetric-padding": _bool_int(manifest.input.symmetric_padding),
        "output-tensor-meta": _bool_int(manifest.deepstream.output_tensor_meta),
        "output-blob-names": manifest.output.name,
    }


def _render_config(values: dict[str, str]) -> str:
    order = [
        "gpu-id",
        "model-engine-file",
        "batch-size",
        "network-mode",
        "network-type",
        "process-mode",
        "gie-unique-id",
        "interval",
        "net-scale-factor",
        "model-color-format",
        "maintain-aspect-ratio",
        "symmetric-padding",
        "output-tensor-meta",
        "output-blob-names",
    ]
    lines = ["[property]"]
    lines.extend(f"{key}={values[key]}" for key in order)
    return "\n".join(lines) + "\n"


def _network_mode(precision: str) -> str:
    normalized = precision.strip().lower()
    if normalized == "fp32":
        return "0"
    if normalized == "int8":
        return "1"
    if normalized == "fp16":
        return "2"
    raise ValueError(f"unsupported DeepStream precision: {precision}")


def _model_color_format(color_format: str) -> str:
    normalized = color_format.strip().upper()
    if normalized == "RGB":
        return "0"
    if normalized == "BGR":
        return "1"
    if normalized in {"GRAY", "GREY"}:
        return "2"
    raise ValueError(f"unsupported DeepStream color format: {color_format}")


def _bool_int(value: bool) -> str:
    return "1" if bool(value) else "0"


def _format_float(value: float) -> str:
    return f"{float(value):.17g}"
