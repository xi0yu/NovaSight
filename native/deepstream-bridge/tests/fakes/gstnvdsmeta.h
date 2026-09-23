#pragma once

#include <gst/gst.h>
#include <nvdsmeta.h>

static inline NvDsBatchMeta* gst_buffer_get_nvds_batch_meta(GstBuffer* buffer) {
    return (NvDsBatchMeta*)buffer->batch_meta;
}
