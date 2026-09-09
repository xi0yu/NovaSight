// Translation-unit-local .cpp that is allowed to pull in <gst/gst.h>.
// nvcc 12.6 mishandles glib-2.0/glib/gmacros.h's legacy __has_attribute()
// gating macros, so we keep all GStreamer access in this g++-compiled
// file and expose the two operations the .cu side uses via C-linkage.
#include "novasight_jetson_preprocess_native_jetson_gst_buffer.h"
#include "novasight_nvbufsurface_payload.h"

#include <gst/gst.h>
#include <nvbufsurface.h>

#include <cstring>

namespace {

struct SurfaceMap {
    GstBuffer* buffer = nullptr;
    GstMapInfo info{};
    bool mapped = false;
};

inline void write_err(char* err_buf, size_t err_buf_size, const char* msg) noexcept {
    if (err_buf == nullptr || err_buf_size == 0) {
        return;
    }
    const size_t length = std::strlen(msg);
    const size_t copy_length = length < err_buf_size - 1 ? length : err_buf_size - 1;
    std::memcpy(err_buf, msg, copy_length);
    err_buf[copy_length] = '\0';
}

int open_surface_impl(
    uint64_t gst_buffer_ptr,
    void** surface_out,
    void** map_owner_out,
    char* err_buf,
    size_t err_buf_size
) {
    if (surface_out == nullptr || map_owner_out == nullptr) {
        write_err(err_buf, err_buf_size, "output pointers are null");
        return NOVASIGHT_GST_RESULT_INVALID_INPUT;
    }
    *surface_out = nullptr;
    *map_owner_out = nullptr;
    if (gst_buffer_ptr == 0) {
        write_err(err_buf, err_buf_size, "gst_buffer_ptr is zero");
        return NOVASIGHT_GST_RESULT_INVALID_INPUT;
    }
    auto* buffer = reinterpret_cast<GstBuffer*>(gst_buffer_ptr);
    if (buffer == nullptr) {
        write_err(err_buf, err_buf_size, "gst_buffer_ptr resolved to null GstBuffer");
        return NOVASIGHT_GST_RESULT_INVALID_INPUT;
    }
    auto* owner = new SurfaceMap();
    owner->buffer = gst_buffer_ref(buffer);
    if (owner->buffer == nullptr) {
        delete owner;
        write_err(err_buf, err_buf_size, "gst_buffer_ref returned null");
        return NOVASIGHT_GST_RESULT_INVALID_INPUT;
    }
    if (!gst_buffer_map(owner->buffer, &owner->info, GST_MAP_READ)) {
        gst_buffer_unref(owner->buffer);
        delete owner;
        write_err(err_buf, err_buf_size, "gst_buffer_map failed for GstBuffer");
        return NOVASIGHT_GST_RESULT_MAP_FAILED;
    }
    owner->mapped = true;
    void* surface = novasight::jetson_preprocess::checked_nvbufsurface_payload(
        owner->info.data,
        owner->info.size,
        sizeof(NvBufSurface)
    );
    if (surface == nullptr) {
        gst_buffer_unmap(owner->buffer, &owner->info);
        gst_buffer_unref(owner->buffer);
        delete owner;
        write_err(err_buf, err_buf_size, "gst_buffer_map did not expose a complete NvBufSurface");
        return NOVASIGHT_GST_RESULT_PAYLOAD_UNAVAILABLE;
    }
    *surface_out = surface;
    *map_owner_out = owner;
    return NOVASIGHT_GST_RESULT_OK;
}

}  // namespace

int novasight_gst_open_surface(
    uint64_t gst_buffer_ptr,
    void** surface_out,
    void** map_owner_out,
    char* err_buf,
    size_t err_buf_size
) {
    try {
        return open_surface_impl(
            gst_buffer_ptr,
            surface_out,
            map_owner_out,
            err_buf,
            err_buf_size
        );
    } catch (...) {
        if (surface_out != nullptr) {
            *surface_out = nullptr;
        }
        if (map_owner_out != nullptr) {
            *map_owner_out = nullptr;
        }
        write_err(err_buf, err_buf_size, "unexpected exception while mapping GstBuffer");
        return NOVASIGHT_GST_RESULT_MAP_FAILED;
    }
}

void novasight_gst_release_surface(void* map_owner) {
    auto* owner = static_cast<SurfaceMap*>(map_owner);
    if (owner == nullptr) {
        return;
    }
    if (owner->mapped && owner->buffer != nullptr) {
        gst_buffer_unmap(owner->buffer, &owner->info);
        owner->mapped = false;
    }
    if (owner->buffer != nullptr) {
        gst_buffer_unref(owner->buffer);
        owner->buffer = nullptr;
    }
    delete owner;
}
