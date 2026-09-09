# NovaSight DeepStream Parser

This library decodes one raw YOLO output tensor into
`NvDsInferObjectDetectionInfo`. DeepStream performs the single class-aware NMS
pass through `cluster-mode=2`; the parser does not run a second NMS.

The supported output layouts are `[C,N]`, `[N,C]`, `[1,C,N]`, and `[1,N,C]`,
where `C` is `4 + classes` (YOLOv8) or `5 + classes` (YOLOv5 objectness).
Coordinates must be pixel-space `cx,cy,w,h` values.

`cargo build -p novasightd --features deepstream` compiles this parser into the
Cargo build directory and embeds its exact path into the daemon automatically.

Build the standalone shared library on the Jetson only for release packaging:

```bash
cmake -S native/deepstream-parser -B build/deepstream-parser \
  -DNOVASIGHT_DEEPSTREAM_ROOT=/opt/nvidia/deepstream/deepstream \
  -DNOVASIGHT_CUDA_ROOT=/usr/local/cuda
cmake --build build/deepstream-parser --parallel
```

The output is `build/deepstream-parser/libnovasight_parser.so`.

Models that already contain Decode/NMS are intentionally rejected by the
current config generator. They require a separate parser contract using
`cluster-mode=4` so NMS is not applied twice.
