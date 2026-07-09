# NovaSight DeepStream Latest Gate

This directory contains the native admission-control work for the
`deepstream_latest` pipeline.

The first committed layer is intentionally GStreamer-free: it is the
single-credit state machine that `nslatestgate` and `nsinferack` will use. This
keeps the admission semantics testable on non-Jetson development machines
before the core is wrapped in real `GstElement`, `GstTask`, and metadata
conversion code.

Current scope:

- single pending latest frame
- single unacked in-flight frame
- ACK-driven credit release
- ACK mismatch and timeout fail-closed behavior
- flush epoch reset so stale ACKs cannot release new-session credit
- EOS drain/drop policy decisions

Build and test:

```bash
cmake -S native/gst-novasight-latest -B build/ns-latest
cmake --build build/ns-latest
ctest --test-dir build/ns-latest --output-on-failure
```

The next layer should add the real elements:

- `nslatestgate`: sink chain stores only the newest pending `GstBuffer`; a
  dispatch task pushes only when the core grants credit.
- `nsinferack`: reads `NsFrameToken` after `nvinfer` and calls ACK on the gate.
- `NsFrameToken` metadata transform: preserves generation across `nvstreammux`
  as `NvDsUserMeta`.
