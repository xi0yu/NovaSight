#ifndef NOVASIGHT_JETSON_PREPROCESS_NATIVE_JETSON_SUPPORT_H
#define NOVASIGHT_JETSON_PREPROCESS_NATIVE_JETSON_SUPPORT_H

#include <stddef.h>

#include <cstdint>
#include <string>
#include <vector>

namespace novasight::jetson_preprocess {

struct TensorRequest {
    uint64_t frame_id = 0;
    uint64_t capture_ts_ns = 0;
    int dmabuf_fd = -1;
    uint64_t gst_buffer_ptr = 0;
    std::string resource_kind;
    std::string resource_memory;
    std::string resource_source;
    std::string resource_pixel_format;
    int resource_width = 0;
    int resource_height = 0;
    std::string pixel_format;
    int width = 0;
    int height = 0;
    int source_width = 0;
    int source_height = 0;
    int roi_offset_x = 0;
    int roi_offset_y = 0;
    bool needs_resize = false;
    std::vector<int> nchw;
    std::string dtype = "float32";
};

struct RequestValidation {
    bool valid = false;
    std::string reason;
    std::string detail;
    TensorRequest request;
};

RequestValidation parse_tensor_request(const char* payload_json);

int dtype_size_bytes(const std::string& dtype);
uint64_t tensor_nbytes(const TensorRequest& request);

bool write_json(char* output, size_t output_size, const std::string& json) noexcept;
bool write_error_json(
    char* output,
    size_t output_size,
    const std::string& reason,
    const std::string& detail
) noexcept;

void write_fallback_json(char* output, size_t output_size, const char* json) noexcept;

std::string unavailable_status_json(
    const std::string& backend,
    const std::string& reason,
    const std::string& detail
);

}  // namespace novasight::jetson_preprocess

#endif
