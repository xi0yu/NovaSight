# NovaSight DeepStream Latest Gate

This directory contains the native admission-control work for the
`deepstream_latest` pipeline.

The first committed layer is the GStreamer-free single-credit state machine
that `nslatestgate` and `nsinferack` use. This keeps the admission semantics
testable on non-Jetson development machines.

The current plugin wrapper is optional at build time. It registers
`nslatestgate` and `nsinferack` when `gstreamer-1.0` and `gstreamer-base-1.0`
development files are available. On machines without those headers, CMake
skips the plugin and still builds the portable core tests.

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

Force plugin compilation on Jetson or any machine with GStreamer development
headers:

```bash
cmake -S native/gst-novasight-latest -B build/ns-latest \
  -DNSLATEST_BUILD_GSTREAMER=ON
cmake --build build/ns-latest
```

Current wrapper scope:

- `nslatestgate`: sink chain stores only the newest pending `GstBuffer`; a
  dispatch thread pushes only when the core grants credit.
- `nsinferack`: reads `NsFrameToken` after `nvinfer` and calls ACK on the gate.
- The wrapper currently uses drop-pending EOS behavior; production drain-latest
  EOS should be added only with an integration test.
- The temporary token transport uses `GstMiniObject` qdata and is valid only
  for same-buffer prototype testing.

The production metadata layer still must add real `GstMeta` plus
`NvDsUserMeta` transform/release callbacks so generation survives
`nvstreammux` and future buffer copies.
