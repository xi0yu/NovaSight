// Translation-unit-local .cpp that is allowed to pull in <gst/gst.h>.
// nvcc 12.6 mishandles glib-2.0/glib/gmacros.h's legacy __has_attribute()
// gating macros, so we keep all GStreamer access in this g++-compiled
// file and expose the two operations the .cu side uses via C-linkage.
#include "novasight_jetson_preprocess_native_jetson_gst_buffer.h"

#include <gst/gst.h>

#include <cstring>
#include <string>

namespace {

struct SurfaceMap {
    GstBuffer* buffer = nullptr;
    GstMapInfo info{};
    bool mapped = false;
};

inline void write_err(char* err_buf, size_t err_buf_size, const std::string& msg) {
    if (err_buf == nullptr || err_buf_size == 0) {
        return;
    }
    std::string truncated = msg;
    if (truncated.size() >= err_buf_size) {
        truncated.resize(err_buf_size - 1);
    }
    std::memcpy(err_buf, truncated.data(), truncated.size());
    err_buf[truncated.size()] = '\0';
}

}  // namespace

int novasight_gst_open_surface(
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
    auto* buffer = reinterpret_cast<GstBuffer*>(static_cast<gpointer>(gst_buffer_ptr));
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
    if (owner->info.data == nullptr || owner->info.size < sizeof(void*)) {
        gst_buffer_unmap(owner->buffer, &owner->info);
        gst_buffer_unref(owner->buffer);
        delete owner;
        write_err(err_buf, err_buf_size, "gst_buffer_map did not expose a usable payload");
        return NOVASIGHT_GST_RESULT_PAYLOAD_UNAVAILABLE;
    }
    *surface_out = *reinterpret_cast<void**>(owner->info.data);
    *map_owner_out = owner;
    return NOVASIGHT_GST_RESULT_OK;
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
