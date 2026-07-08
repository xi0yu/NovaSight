#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from novasight.model_registry.import_model import import_onnx_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Import an ONNX model into NovaSight originals.")
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models/originals"))
    parser.add_argument("--model-id", default="")
    parser.add_argument("--display-name", default="")
    parser.add_argument("--classes", default="", help="Comma-separated class names.")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--precision", default="fp16", choices=("fp32", "fp16", "int8"))
    args = parser.parse_args()

    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    imported = import_onnx_model(
        args.onnx_path,
        output_dir=args.output_dir,
        model_id=args.model_id or None,
        display_name=args.display_name or None,
        class_names=classes or None,
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        runtime_precision=args.precision,
    )
    print(
        json.dumps(
            {
                "model_id": imported.manifest.model_id,
                "model_path": str(imported.model_path),
                "manifest_path": str(imported.manifest_path),
                "input": imported.manifest.input.__dict__,
                "output": imported.manifest.output.__dict__,
                "class_count": imported.manifest.output.class_count,
                "model_fingerprint": imported.manifest.model_fingerprint,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
