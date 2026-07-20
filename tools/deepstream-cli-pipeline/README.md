# deepstream-cli-pipeline

Small Rust CLI for building and optionally running the NovaSight DeepStream
command-line pipeline on Jetson.

It does not use `gstreamer-rs`; it prints or spawns `gst-launch-1.0` directly.
That keeps the demo useful on machines that have the runtime plugins but not the
GStreamer development headers.

```bash
cargo run --manifest-path tools/deepstream-cli-pipeline/Cargo.toml -- \
  --run --verbose --fps-display --timeout-seconds 30 \
  --device /dev/video0 --format MJPG --width 1920 --height 1080 --fps 120 \
  --roi-left 640 --roi-top 220 --roi-size 640 \
  --model-width 640 --model-height 640 \
  --nvinfer-config data/runtime/deepstream/active-nvinfer.ini
```

Without `--run`, the tool only prints the command. The generated path is:

```text
v4l2src -> latest queue -> nvv4l2decoder -> NVMM
-> latest queue -> nvvidconv ROI+resize -> NVMM NV12
-> latest queue -> nvstreammux -> nvinfer -> fakesink/fpsdisplaysink
```

If Rust is not installed on the Jetson, use the shell wrapper instead:

```bash
scripts/run_deepstream_gst_pipeline.sh
```

Override the defaults with environment variables:

```bash
DEVICE=/dev/video0 FORMAT=MJPG WIDTH=1920 HEIGHT=1080 FPS=120 \
ROI_LEFT=640 ROI_TOP=220 ROI_SIZE=640 \
MODEL_WIDTH=640 MODEL_HEIGHT=640 \
NVINFER_CONFIG=data/runtime/deepstream/active-nvinfer.ini \
TIMEOUT_SECONDS=30 scripts/run_deepstream_gst_pipeline.sh
```
