#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _GstBuffer GstBuffer;

#define NS_DS_BRIDGE_ABI_VERSION 1U
#define NS_DS_MAX_DETECTIONS 256U
#define NS_DS_ANY_SOURCE UINT32_MAX

typedef enum NsDsBridgeStatus {
    NS_DS_BRIDGE_OK = 0,
    NS_DS_BRIDGE_INVALID_ARGUMENT = 1,
    NS_DS_BRIDGE_OUTPUT_TOO_SMALL = 2,
    NS_DS_BRIDGE_NO_BATCH_META = 3,
    NS_DS_BRIDGE_NO_FRAME_META = 4,
    NS_DS_BRIDGE_SOURCE_NOT_FOUND = 5,
    NS_DS_BRIDGE_INTERNAL_ERROR = 6,
} NsDsBridgeStatus;

typedef enum NsDsFrameFlags {
    NS_DS_FRAME_INFERENCE_DONE = 1U << 0U,
    NS_DS_FRAME_DETECTIONS_TRUNCATED = 1U << 1U,
    NS_DS_FRAME_BUFFER_PTS_VALID = 1U << 2U,
    NS_DS_FRAME_META_PTS_VALID = 1U << 3U,
    NS_DS_FRAME_HAS_UNTRACKED_OBJECT = 1U << 4U,
} NsDsFrameFlags;

typedef struct NsDsDetection {
    uint64_t object_id;
    float left;
    float top;
    float width;
    float height;
    float confidence;
    int32_t class_id;
    int32_t component_id;
    uint32_t flags;
} NsDsDetection;

typedef struct NsDsFrameSnapshot {
    uint32_t abi_version;
    uint32_t struct_size;
    int64_t frame_num;
    uint64_t buffer_pts_ns;
    uint64_t frame_pts_ns;
    uint64_t ntp_timestamp_ns;
    uint32_t source_id;
    uint32_t batch_id;
    uint32_t source_width;
    uint32_t source_height;
    uint32_t pipeline_width;
    uint32_t pipeline_height;
    uint32_t detection_count;
    uint32_t truncated_count;
    uint32_t invalid_object_count;
    uint32_t flags;
    NsDsDetection detections[NS_DS_MAX_DETECTIONS];
} NsDsFrameSnapshot;

uint32_t ns_ds_bridge_abi_version(void);
uint32_t ns_ds_bridge_frame_snapshot_size(void);

/*
 * Borrow buffer only for this call and copy the selected frame metadata into
 * caller-owned storage. No GstBuffer, NvDsBatchMeta, frame-meta, object-meta,
 * or list pointer is retained after this function returns.
 */
int32_t ns_ds_extract_frame(
    GstBuffer* buffer,
    uint32_t source_id,
    NsDsFrameSnapshot* output,
    uint32_t output_size);

const char* ns_ds_bridge_status_string(int32_t status);

#ifdef __cplusplus
}
#endif
