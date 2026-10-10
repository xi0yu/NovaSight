#ifndef NOVASIGHT_JETSON_PREPROCESS_NATIVE_JETSON_GST_BUFFER_H
#define NOVASIGHT_JETSON_PREPROCESS_NATIVE_JETSON_GST_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Result codes for gst surface resolver helpers.
typedef enum {
    NOVASIGHT_GST_RESULT_OK = 0,
    NOVASIGHT_GST_RESULT_INVALID_INPUT = 1,
    NOVASIGHT_GST_RESULT_MAP_FAILED = 2,
    NOVASIGHT_GST_RESULT_PAYLOAD_UNAVAILABLE = 3,
} novasight_gst_result;

// gst_buffer_ptr entry path. Implemented in
// novasight_jetson_preprocess_native_jetson_gst_buffer.cpp using the
// official GStreamer headers — that file is compiled with g++, not nvcc,
// so the legacy __has_attribute() / gmacros.h preprocessing problem does
// not affect it.
//
// In: gst_buffer_ptr is the integer pointer value borrowed from the live
// GstBuffer held by the caller's frame lease.
// Out: *surface_out receives a borrowed NvBufSurface* if the call
// succeeds, or nullptr on failure. *map_owner_out is an opaque handle that
// the caller must release via novasight_gst_release_surface; on success
// map_owner_out is non-null. err_buf receives a short human-readable
// description on failure (otherwise empty).
int novasight_gst_open_surface(
    uint64_t gst_buffer_ptr,
    void** surface_out,
    void** map_owner_out,
    char* err_buf,
    size_t err_buf_size
);

void novasight_gst_release_surface(void* map_owner);

#ifdef __cplusplus
}
#endif

#endif  // NOVASIGHT_JETSON_PREPROCESS_NATIVE_JETSON_GST_BUFFER_H
