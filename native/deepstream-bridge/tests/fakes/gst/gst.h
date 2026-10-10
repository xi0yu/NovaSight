#pragma once

#include <stdint.h>

typedef uint64_t GstClockTime;

typedef struct _GstBuffer {
    uint64_t pts;
    int pts_valid;
    void* batch_meta;
} GstBuffer;

#define GST_CLOCK_TIME_NONE UINT64_MAX
#define GST_BUFFER_PTS(buffer) ((buffer)->pts)
#define GST_BUFFER_PTS_IS_VALID(buffer) ((buffer)->pts_valid != 0)
