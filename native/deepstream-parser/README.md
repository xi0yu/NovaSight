# NovaSight DeepStream Parser

This library decodes one raw YOLO output tensor into
`NvDsInferObjectDetectionInfo`. DeepStream performs the single class-aware NMS
pass through `cluster-mode=2`; the parser does not run a second NMS.

The supported output layouts are `[C,N]`, `[N,C]`, `[1,C,N]`, and `[1,N,C]`,
where `C` is `4 + classes` (YOLOv8) or `5 + classes` (YOLOv5 objectness).
Coordinates must be pixel-space `cx,cy,w,h` values.

Build on the Jetson with its installed DeepStream headers:

```bash
cmake -S native/deepstream-parser -B build/deepstream-parser \
  -DNOVASIGHT_DEEPSTREAM_ROOT=/opt/nvidia/deepstream/deepstream-7.1
cmake --build build/deepstream-parser --parallel
```

The output is `build/deepstream-parser/libnovasight_parser.so`.

Models that already contain Decode/NMS are intentionally rejected by the
current config generator. They require a separate parser contract using
`cluster-mode=4` so NMS is not applied twice.
