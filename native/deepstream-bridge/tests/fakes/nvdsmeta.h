#pragma once

#include <stdint.h>

typedef int gboolean;

#ifndef FALSE
#define FALSE 0
#endif

#define UNTRACKED_OBJECT_ID UINT64_MAX

typedef struct _GList {
    void* data;
    struct _GList* next;
} GList;

typedef GList NvDsMetaList;

typedef struct _NvOSD_RectParams {
    float left;
    float top;
    float width;
    float height;
} NvOSD_RectParams;

typedef struct _NvDsObjectMeta {
    int32_t unique_component_id;
    int32_t class_id;
    uint64_t object_id;
    float confidence;
    NvOSD_RectParams rect_params;
} NvDsObjectMeta;

typedef struct _NvDsFrameMeta {
    uint32_t batch_id;
    int32_t frame_num;
    uint64_t buf_pts;
    uint64_t ntp_timestamp;
    uint32_t source_id;
    uint32_t source_frame_width;
    uint32_t source_frame_height;
    gboolean bInferDone;
    NvDsMetaList* obj_meta_list;
    uint32_t pipeline_width;
    uint32_t pipeline_height;
} NvDsFrameMeta;

typedef struct _NvDsBatchMeta {
    NvDsMetaList* frame_meta_list;
} NvDsBatchMeta;
