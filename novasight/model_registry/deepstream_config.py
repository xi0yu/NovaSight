from __future__ import annotations

from pathlib import Path

from .fingerprint import sha256_text, stable_json
from .manifest import ModelManifest


DEEPSTREAM_CONFIG_TEMPLATE_VERSION = "nvinfer-template-v1"
DEEPSTREAM_CONFIG_FINGERPRINT_KEY = "novasight-config-fingerprint"


def generate_nvinfer_config(
    manifest: ModelManifest,
    *,
    engine_path: Path,
) -> tuple[str, str]:
    config = _config_payload(manifest, engine_path=Path(engine_path).resolve())
    fingerprint = nvinfer_config_fingerprint(manifest)
    text = _render_config(config, fingerprint=fingerprint)
    return text, fingerprint


def nvinfer_config_fingerprint(manifest: ModelManifest) -> str:
    fingerprint_payload = {
        "template": DEEPSTREAM_CONFIG_TEMPLATE_VERSION,
        "model_fingerprint": manifest.model_fingerprint,
        "runtime": manifest.runtime.__dict__,
        "input": manifest.input.__dict__,
        "output": manifest.output.__dict__,
        "deepstream": manifest.deepstream.__dict__,
    }
    return f"{DEEPSTREAM_CONFIG_TEMPLATE_VERSION}:{sha256_text(stable_json(fingerprint_payload))}"


def read_nvinfer_config_fingerprint(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    prefix = f"# {DEEPSTREAM_CONFIG_FINGERPRINT_KEY}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def read_nvinfer_config_property(path: Path, key: str) -> str:
    expected = str(key).strip()
    if not expected:
        return ""
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == expected:
            return _unquote_config_value(value.strip())
    return ""


def validate_nvinfer_config_engine_path(path: Path, expected_engine_path: Path) -> None:
    actual = read_nvinfer_config_property(path, "model-engine-file")
    if not actual:
        raise ValueError("DeepStream nvinfer config missing model-engine-file")
    expected = Path(expected_engine_path).expanduser().resolve(strict=False)
    actual_path = Path(actual).expanduser()
    if not actual_path.is_absolute():
        actual_path = Path(path).parent / actual_path
    actual_resolved = actual_path.resolve(strict=False)
    if actual_resolved != expected:
        raise ValueError(
            "DeepStream nvinfer config model-engine-file does not match artifact "
            f"(expected={expected}, actual={actual_resolved})"
        )


def validate_nvinfer_config_properties(path: Path, manifest: ModelManifest) -> None:
    expected = _config_payload(
        manifest,
        engine_path=Path(read_nvinfer_config_property(path, "model-engine-file") or "model.engine"),
    )
    checked_keys = (
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
    )
    mismatches: list[str] = []
    for key in checked_keys:
        actual = read_nvinfer_config_property(path, key)
        if not actual:
            mismatches.append(f"{key}=missing expected {expected[key]}")
            continue
        if actual != expected[key]:
            mismatches.append(f"{key}={actual} expected {expected[key]}")
    if mismatches:
        raise ValueError(
            "DeepStream nvinfer config properties do not match model manifest: "
            + "; ".join(mismatches)
        )


def _config_payload(manifest: ModelManifest, *, engine_path: Path) -> dict[str, str]:
    return {
        "input-contract": _tensor_contract(
            manifest.input.name,
            manifest.input.shape,
            manifest.input.dtype,
            manifest.input.layout,
        ),
        "output-contract": _tensor_contract(
            manifest.output.name,
            manifest.output.shape,
            manifest.output.dtype,
            manifest.output.layout,
        ),
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


def _unquote_config_value(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def _render_config(values: dict[str, str], *, fingerprint: str) -> str:
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
    lines = [
        f"# {DEEPSTREAM_CONFIG_FINGERPRINT_KEY}={fingerprint}",
        f"# novasight-model-input={values['input-contract']}",
        f"# novasight-model-output={values['output-contract']}",
        "[property]",
    ]
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


def _tensor_contract(name: str, shape: list[int], dtype: str, layout: str) -> str:
    dims = "x".join(str(dim) for dim in shape)
    return f"{name} {dims} {dtype} {layout}"
