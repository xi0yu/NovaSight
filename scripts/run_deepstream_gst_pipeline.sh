#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

GST_LAUNCH="${GST_LAUNCH:-gst-launch-1.0}"
DEVICE="${DEVICE:-/dev/video0}"
FORMAT="${FORMAT:-MJPG}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"
FPS="${FPS:-120}"
IO_MODE="${IO_MODE:-2}"
ROI_LEFT="${ROI_LEFT:-640}"
ROI_TOP="${ROI_TOP:-220}"
ROI_SIZE="${ROI_SIZE:-640}"
ROI_WIDTH="${ROI_WIDTH:-${ROI_SIZE}}"
ROI_HEIGHT="${ROI_HEIGHT:-${ROI_SIZE}}"
MODEL_WIDTH="${MODEL_WIDTH:-640}"
MODEL_HEIGHT="${MODEL_HEIGHT:-640}"
NVINFER_CONFIG="${NVINFER_CONFIG:-${ROOT_DIR}/data/runtime/deepstream/active-nvinfer.ini}"
BATCHED_PUSH_TIMEOUT_US="${BATCHED_PUSH_TIMEOUT_US:-0}"
SINK="${SINK:-fpsdisplaysink video-sink=fakesink text-overlay=false sync=false signal-fps-measurements=false}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-30}"
VERBOSE="${VERBOSE:-1}"
FORMAT_UPPER="$(printf '%s' "${FORMAT}" | tr '[:lower:]' '[:upper:]')"
VERBOSE_LOWER="$(printf '%s' "${VERBOSE}" | tr '[:upper:]' '[:lower:]')"

case "${FORMAT_UPPER}" in
  MJPG|MJPEG)
    CAPTURE_CAPS="image/jpeg,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1"
    DECODE_PATH="queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! jpegparse ! nvv4l2decoder mjpeg=1 ! video/x-raw(memory:NVMM),format=I420 ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
    ;;
  NV12)
    CAPTURE_CAPS="video/x-raw,format=NV12,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1"
    DECODE_PATH="queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
    ;;
  YUYV|YUY2)
    CAPTURE_CAPS="video/x-raw,format=YUY2,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1"
    DECODE_PATH="queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
    ;;
  *)
    echo "FORMAT must be MJPG, MJPEG, NV12, YUYV, or YUY2; got ${FORMAT}" >&2
    exit 2
    ;;
esac

ROI_RIGHT=$((ROI_LEFT + ROI_WIDTH))
ROI_BOTTOM=$((ROI_TOP + ROI_HEIGHT))
if (( WIDTH <= 0 || HEIGHT <= 0 || FPS <= 0 || ROI_WIDTH <= 0 || ROI_HEIGHT <= 0 || MODEL_WIDTH <= 0 || MODEL_HEIGHT <= 0 )); then
  echo "width/height/fps/roi/model dimensions must be positive" >&2
  exit 2
fi
if (( ROI_RIGHT > WIDTH || ROI_BOTTOM > HEIGHT )); then
  echo "ROI must stay inside capture frame: right=${ROI_RIGHT}/${WIDTH} bottom=${ROI_BOTTOM}/${HEIGHT}" >&2
  exit 2
fi

PIPELINE=(
  "${GST_LAUNCH}"
)
if [[ "${VERBOSE}" == "1" || "${VERBOSE_LOWER}" == "true" ]]; then
  PIPELINE+=("-v")
fi
PIPELINE+=(
  v4l2src "name=capture-source" "device=${DEVICE}" "io-mode=${IO_MODE}" do-timestamp=true
  !
  "${CAPTURE_CAPS}"
  !
  ${DECODE_PATH}
  !
  nvvidconv "left=${ROI_LEFT}" "right=${ROI_RIGHT}" "top=${ROI_TOP}" "bottom=${ROI_BOTTOM}"
  !
  "video/x-raw(memory:NVMM),format=NV12,width=${MODEL_WIDTH},height=${MODEL_HEIGHT},pixel-aspect-ratio=1/1"
  !
  queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream
  !
  mux.sink_0
  nvstreammux name=mux batch-size=1 live-source=1 "width=${MODEL_WIDTH}" "height=${MODEL_HEIGHT}" sync-inputs=0 "batched-push-timeout=${BATCHED_PUSH_TIMEOUT_US}"
  !
  nvinfer name=primary-infer "config-file-path=${NVINFER_CONFIG}" batch-size=1
  !
  ${SINK}
  "$@"
)

printf 'command:'
printf ' %q' "${PIPELINE[@]}"
printf '\n'

if [[ "${TIMEOUT_SECONDS}" == "0" ]]; then
  exec "${PIPELINE[@]}"
fi

if command -v timeout >/dev/null 2>&1; then
  exec timeout "${TIMEOUT_SECONDS}" "${PIPELINE[@]}"
fi

"${PIPELINE[@]}" &
child=$!
sleep "${TIMEOUT_SECONDS}"
kill "${child}" >/dev/null 2>&1 || true
wait "${child}" >/dev/null 2>&1 || true
