# NovaSight DeepStream bridge

This module is the only seam where the Rust runtime sees the NVIDIA DeepStream
ABI. It synchronously copies one frame's metadata into caller-owned,
fixed-layout storage.

The ownership rule is strict: `GstBuffer`, `NvDsBatchMeta`, `NvDsFrameMeta`,
`NvDsObjectMeta`, and their `GList` nodes are borrowed only while
`ns_ds_extract_frame()` is executing. No NVIDIA pointer is retained or returned.

The Rust pad probe should claim a preallocated write slot, call
`ns_ds_extract_frame()` directly into that slot, then publish the slot's
generation with release ordering. The consumer reads only the owned
`NsDsFrameSnapshot` after acquire ordering. The producer must drop the incoming
frame instead of waiting when no write slot is available.

`cargo build -p novasightd --features deepstream` compiles and statically links
this bridge automatically. No prebuilt bridge or `LD_LIBRARY_PATH` is required
for source-tree development.

Build the standalone shared library on Jetson only for ABI testing or release
packaging:

```bash
cmake -S native/deepstream-bridge -B build/deepstream-bridge \
  -DNOVASIGHT_DEEPSTREAM_ROOT=/opt/nvidia/deepstream/deepstream
cmake --build build/deepstream-bridge --parallel
ctest --test-dir build/deepstream-bridge --output-on-failure
```

The output is `build/deepstream-bridge/libnovasight_deepstream_bridge.so`.
The load test links that library against the installed DeepStream SDK and checks
the ABI version and frame layout before the Rust runtime is started.
