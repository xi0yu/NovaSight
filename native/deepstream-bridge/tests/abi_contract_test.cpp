#include "novasight_deepstream_bridge.h"

#include <cstddef>

static_assert(NS_DS_BRIDGE_ABI_VERSION == 1U);
static_assert(NS_DS_MAX_DETECTIONS == 256U);
static_assert(sizeof(NsDsDetection) == 40U);
static_assert(alignof(NsDsDetection) == 8U);
static_assert(offsetof(NsDsDetection, left) == 8U);
static_assert(offsetof(NsDsDetection, confidence) == 24U);
static_assert(offsetof(NsDsDetection, class_id) == 28U);
static_assert(offsetof(NsDsDetection, flags) == 36U);
static_assert(sizeof(NsDsFrameSnapshot) == 10320U);
static_assert(alignof(NsDsFrameSnapshot) == 8U);
static_assert(offsetof(NsDsFrameSnapshot, frame_num) == 8U);
static_assert(offsetof(NsDsFrameSnapshot, buffer_pts_ns) == 16U);
static_assert(offsetof(NsDsFrameSnapshot, source_id) == 40U);
static_assert(offsetof(NsDsFrameSnapshot, detection_count) == 64U);
static_assert(offsetof(NsDsFrameSnapshot, detections) == 80U);

int main() {
    NsDsFrameSnapshot frame{};
    return frame.detection_count == 0U ? 0 : 1;
}
