#include "novasight_deepstream_bridge.h"

#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>

#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>

namespace {

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

void initialize_output(NsDsFrameSnapshot* output) {
    std::memset(output, 0, sizeof(*output));
    output->abi_version = NS_DS_BRIDGE_ABI_VERSION;
    output->struct_size = static_cast<uint32_t>(sizeof(*output));
}

NvDsFrameMeta* select_frame(NvDsBatchMeta* batch_meta, uint32_t source_id) {
    for (NvDsMetaList* node = batch_meta->frame_meta_list; node != nullptr; node = node->next) {
        auto* frame_meta = static_cast<NvDsFrameMeta*>(node->data);
        if (frame_meta == nullptr) {
            continue;
        }
        if (source_id == NS_DS_ANY_SOURCE || frame_meta->source_id == source_id) {
            return frame_meta;
        }
    }
    return nullptr;
}

void copy_frame(
    GstBuffer* buffer,
    const NvDsFrameMeta& frame_meta,
    NsDsFrameSnapshot* output) {
    output->frame_num = static_cast<int64_t>(frame_meta.frame_num);
    output->frame_pts_ns = static_cast<uint64_t>(frame_meta.buf_pts);
    output->ntp_timestamp_ns = static_cast<uint64_t>(frame_meta.ntp_timestamp);
    output->source_id = static_cast<uint32_t>(frame_meta.source_id);
    output->batch_id = static_cast<uint32_t>(frame_meta.batch_id);
    output->source_width = static_cast<uint32_t>(frame_meta.source_frame_width);
    output->source_height = static_cast<uint32_t>(frame_meta.source_frame_height);
    output->pipeline_width = static_cast<uint32_t>(frame_meta.pipeline_width);
    output->pipeline_height = static_cast<uint32_t>(frame_meta.pipeline_height);

    if (frame_meta.bInferDone != FALSE) {
        output->flags |= NS_DS_FRAME_INFERENCE_DONE;
    }
    if (GST_BUFFER_PTS_IS_VALID(buffer)) {
        output->buffer_pts_ns = static_cast<uint64_t>(GST_BUFFER_PTS(buffer));
        output->flags |= NS_DS_FRAME_BUFFER_PTS_VALID;
    }
    if (frame_meta.buf_pts != GST_CLOCK_TIME_NONE) {
        output->flags |= NS_DS_FRAME_META_PTS_VALID;
    }

    uint32_t observed = 0U;
    for (NvDsMetaList* node = frame_meta.obj_meta_list; node != nullptr; node = node->next) {
        auto* object_meta = static_cast<NvDsObjectMeta*>(node->data);
        if (object_meta == nullptr) {
            ++output->invalid_object_count;
            continue;
        }
        ++observed;
        if (output->detection_count >= NS_DS_MAX_DETECTIONS) {
            ++output->truncated_count;
            continue;
        }

        NsDsDetection& detection = output->detections[output->detection_count++];
        detection.object_id = static_cast<uint64_t>(object_meta->object_id);
        detection.left = static_cast<float>(object_meta->rect_params.left);
        detection.top = static_cast<float>(object_meta->rect_params.top);
        detection.width = static_cast<float>(object_meta->rect_params.width);
        detection.height = static_cast<float>(object_meta->rect_params.height);
        detection.confidence = static_cast<float>(object_meta->confidence);
        detection.class_id = static_cast<int32_t>(object_meta->class_id);
        detection.component_id = static_cast<int32_t>(object_meta->unique_component_id);
        if (object_meta->object_id == UNTRACKED_OBJECT_ID) {
            output->flags |= NS_DS_FRAME_HAS_UNTRACKED_OBJECT;
        }
    }

    if (observed > NS_DS_MAX_DETECTIONS) {
        output->flags |= NS_DS_FRAME_DETECTIONS_TRUNCATED;
    }
}

}  // namespace

extern "C" uint32_t ns_ds_bridge_abi_version(void) {
    return NS_DS_BRIDGE_ABI_VERSION;
}

extern "C" uint32_t ns_ds_bridge_frame_snapshot_size(void) {
    return static_cast<uint32_t>(sizeof(NsDsFrameSnapshot));
}

extern "C" int32_t ns_ds_extract_frame(
    GstBuffer* buffer,
    uint32_t source_id,
    NsDsFrameSnapshot* output,
    uint32_t output_size) {
    if (buffer == nullptr || output == nullptr) {
        return NS_DS_BRIDGE_INVALID_ARGUMENT;
    }
    if (output_size < sizeof(NsDsFrameSnapshot)) {
        return NS_DS_BRIDGE_OUTPUT_TOO_SMALL;
    }

    initialize_output(output);
    try {
        NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
        if (batch_meta == nullptr) {
            return NS_DS_BRIDGE_NO_BATCH_META;
        }
        if (batch_meta->frame_meta_list == nullptr) {
            return NS_DS_BRIDGE_NO_FRAME_META;
        }
        NvDsFrameMeta* frame_meta = select_frame(batch_meta, source_id);
        if (frame_meta == nullptr) {
            return NS_DS_BRIDGE_SOURCE_NOT_FOUND;
        }
        copy_frame(buffer, *frame_meta, output);
        return NS_DS_BRIDGE_OK;
    } catch (const std::exception&) {
        initialize_output(output);
        return NS_DS_BRIDGE_INTERNAL_ERROR;
    } catch (...) {
        initialize_output(output);
        return NS_DS_BRIDGE_INTERNAL_ERROR;
    }
}

extern "C" const char* ns_ds_bridge_status_string(int32_t status) {
    switch (status) {
        case NS_DS_BRIDGE_OK:
            return "ok";
        case NS_DS_BRIDGE_INVALID_ARGUMENT:
            return "invalid argument";
        case NS_DS_BRIDGE_OUTPUT_TOO_SMALL:
            return "output buffer is smaller than NsDsFrameSnapshot";
        case NS_DS_BRIDGE_NO_BATCH_META:
            return "GstBuffer has no NvDsBatchMeta";
        case NS_DS_BRIDGE_NO_FRAME_META:
            return "NvDsBatchMeta has no frame metadata";
        case NS_DS_BRIDGE_SOURCE_NOT_FOUND:
            return "requested source_id is not present in NvDsBatchMeta";
        case NS_DS_BRIDGE_INTERNAL_ERROR:
            return "DeepStream metadata extraction failed";
        default:
            return "unknown DeepStream bridge status";
    }
}
