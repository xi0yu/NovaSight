#include "novasight_deepstream_bridge.h"

#include <gst/gst.h>
#include <nvdsmeta.h>

#include <cmath>

namespace {

bool close(float left, float right) {
    return std::fabs(left - right) <= 0.0001F;
}

}  // namespace

int main() {
    NvDsObjectMeta first{};
    first.unique_component_id = 7;
    first.class_id = 2;
    first.object_id = 42U;
    first.confidence = 0.75F;
    first.rect_params = {10.0F, 20.0F, 30.0F, 40.0F};

    NvDsObjectMeta second{};
    second.unique_component_id = 7;
    second.class_id = 1;
    second.object_id = UNTRACKED_OBJECT_ID;
    second.confidence = 0.5F;
    second.rect_params = {50.0F, 60.0F, 70.0F, 80.0F};

    NvDsMetaList second_node{&second, nullptr};
    NvDsMetaList first_node{&first, &second_node};
    NvDsFrameMeta frame{};
    frame.batch_id = 0U;
    frame.frame_num = 123;
    frame.buf_pts = 1'000'000U;
    frame.ntp_timestamp = 2'000'000U;
    frame.source_id = 9U;
    frame.source_frame_width = 1920U;
    frame.source_frame_height = 1080U;
    frame.bInferDone = 1;
    frame.obj_meta_list = &first_node;
    frame.pipeline_width = 640U;
    frame.pipeline_height = 640U;

    NvDsMetaList frame_node{&frame, nullptr};
    NvDsBatchMeta batch{&frame_node};
    GstBuffer buffer{900'000U, 1, &batch};
    NsDsFrameSnapshot output{};

    if (ns_ds_extract_frame(&buffer, 9U, &output, sizeof(output)) != NS_DS_BRIDGE_OK) {
        return 1;
    }
    if (output.abi_version != NS_DS_BRIDGE_ABI_VERSION
        || output.struct_size != sizeof(output)
        || output.frame_num != 123
        || output.buffer_pts_ns != 900'000U
        || output.frame_pts_ns != 1'000'000U
        || output.source_id != 9U
        || output.source_width != 1920U
        || output.pipeline_width != 640U
        || output.detection_count != 2U) {
        return 2;
    }
    if (!close(output.detections[0].left, 10.0F)
        || !close(output.detections[0].confidence, 0.75F)
        || output.detections[0].class_id != 2
        || output.detections[0].object_id != 42U) {
        return 3;
    }
    if ((output.flags & NS_DS_FRAME_INFERENCE_DONE) == 0U
        || (output.flags & NS_DS_FRAME_BUFFER_PTS_VALID) == 0U
        || (output.flags & NS_DS_FRAME_META_PTS_VALID) == 0U
        || (output.flags & NS_DS_FRAME_HAS_UNTRACKED_OBJECT) == 0U) {
        return 4;
    }
    if (ns_ds_extract_frame(&buffer, 8U, &output, sizeof(output))
        != NS_DS_BRIDGE_SOURCE_NOT_FOUND) {
        return 5;
    }
    if (ns_ds_extract_frame(&buffer, 9U, &output, sizeof(output) - 1U)
        != NS_DS_BRIDGE_OUTPUT_TOO_SMALL) {
        return 6;
    }
    buffer.batch_meta = nullptr;
    if (ns_ds_extract_frame(&buffer, NS_DS_ANY_SOURCE, &output, sizeof(output))
        != NS_DS_BRIDGE_NO_BATCH_META) {
        return 7;
    }
    return 0;
}
