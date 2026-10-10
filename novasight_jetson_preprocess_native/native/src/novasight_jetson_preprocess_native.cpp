#include "novasight_jetson_preprocess_native.h"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct PayloadValidation {
    bool valid = false;
    std::string reason;
    std::string detail;
};

void write_json(char* output, size_t output_size, const char* json) noexcept {
    if (output == nullptr || output_size == 0) {
        return;
    }
    if (json == nullptr) {
        json = "";
    }
    const size_t length = std::strlen(json);
    const size_t copy_length = length < output_size - 1 ? length : output_size - 1;
    std::memcpy(output, json, copy_length);
    output[copy_length] = '\0';
}

void write_error_json(
    char* output,
    size_t output_size,
    const std::string& reason,
    const std::string& detail
) {
    const std::string json =
        "{\"reason\":\"" + reason + "\",\"detail\":\"" + detail + "\"}";
    write_json(output, output_size, json.c_str());
}

const char* field_value_start(const std::string& json, const char* key) {
    const std::string key_text(key);
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    const char* string_start = nullptr;
    const char* cursor = json.c_str();
    while (*cursor != '\0') {
        const char ch = *cursor;
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
                const std::string field_name(
                    string_start,
                    static_cast<size_t>(cursor - string_start)
                );
                if (depth == 1 && field_name == key_text) {
                    const char* value = cursor + 1;
                    while (*value != '\0'
                           && std::isspace(static_cast<unsigned char>(*value)) != 0) {
                        ++value;
                    }
                    if (*value != ':') {
                        return nullptr;
                    }
                    ++value;
                    while (*value != '\0'
                           && std::isspace(static_cast<unsigned char>(*value)) != 0) {
                        ++value;
                    }
                    return value;
                }
            }
            ++cursor;
            continue;
        }
        if (ch == '"') {
            in_string = true;
            string_start = cursor + 1;
        } else if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
        }
        ++cursor;
    }
    return nullptr;
}

bool json_value_terminator(char ch) {
    return ch == '\0'
           || ch == ','
           || ch == '}'
           || ch == ']'
           || std::isspace(static_cast<unsigned char>(ch)) != 0;
}

bool read_string_field(const std::string& json, const char* key, std::string* value) {
    const char* cursor = field_value_start(json, key);
    if (cursor == nullptr || *cursor != '"') {
        return false;
    }
    ++cursor;
    std::string result;
    while (*cursor != '\0' && *cursor != '"') {
        if (*cursor == '\\' && cursor[1] != '\0') {
            ++cursor;
        }
        result.push_back(*cursor);
        ++cursor;
    }
    if (*cursor != '"' || !json_value_terminator(cursor[1])) {
        return false;
    }
    *value = result;
    return true;
}

bool read_int_field(const std::string& json, const char* key, long long* value) {
    const char* cursor = field_value_start(json, key);
    if (cursor == nullptr || std::strncmp(cursor, "null", 4) == 0) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(cursor, &end, 10);
    if (end == cursor || errno == ERANGE || !json_value_terminator(*end)) {
        return false;
    }
    *value = parsed;
    return true;
}

bool read_bool_field(const std::string& json, const char* key, bool* value) {
    const char* cursor = field_value_start(json, key);
    if (cursor == nullptr) {
        return false;
    }
    if (std::strncmp(cursor, "true", 4) == 0 && json_value_terminator(cursor[4])) {
        *value = true;
        return true;
    }
    if (std::strncmp(cursor, "false", 5) == 0 && json_value_terminator(cursor[5])) {
        *value = false;
        return true;
    }
    return false;
}

bool read_object_field(
    const std::string& json,
    const char* key,
    std::string* object_json
) {
    const char* cursor = field_value_start(json, key);
    if (cursor == nullptr || *cursor != '{') {
        return false;
    }
    const char* start = cursor;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    while (*cursor != '\0') {
        const char ch = *cursor;
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            ++cursor;
            continue;
        }
        if (ch == '"') {
            in_string = true;
        } else if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) {
                if (!json_value_terminator(cursor[1])) {
                    return false;
                }
                object_json->assign(start, static_cast<size_t>(cursor - start + 1));
                return true;
            }
        }
        ++cursor;
    }
    return false;
}

bool read_int_array_field(
    const std::string& json,
    const char* key,
    std::vector<long long>* values
) {
    const char* cursor = field_value_start(json, key);
    if (cursor == nullptr || *cursor != '[') {
        return false;
    }
    ++cursor;
    std::vector<long long> parsed_values;
    bool expect_value = true;
    while (*cursor != '\0') {
        while (*cursor != '\0' && std::isspace(static_cast<unsigned char>(*cursor)) != 0) {
            ++cursor;
        }
        if (*cursor == ']') {
            if (expect_value && !parsed_values.empty()) {
                return false;
            }
            break;
        }
        if (!expect_value) {
            if (*cursor != ',') {
                return false;
            }
            ++cursor;
            while (*cursor != '\0' && std::isspace(static_cast<unsigned char>(*cursor)) != 0) {
                ++cursor;
            }
            if (*cursor == ']') {
                return false;
            }
        }
        char* end = nullptr;
        errno = 0;
        const long long parsed = std::strtoll(cursor, &end, 10);
        if (end == cursor || errno == ERANGE || !json_value_terminator(*end)) {
            return false;
        }
        parsed_values.push_back(parsed);
        cursor = end;
        expect_value = false;
    }
    if (*cursor != ']' || !json_value_terminator(cursor[1])) {
        return false;
    }
    *values = parsed_values;
    return true;
}

PayloadValidation validate_payload(const char* payload_json) {
    if (payload_json == nullptr || payload_json[0] == '\0') {
        return {false, "payload_required", "payload_json is empty"};
    }
    const std::string json(payload_json);
    long long frame_id = 0;
    if (!read_int_field(json, "frame_id", &frame_id) || frame_id <= 0) {
        return {
            false,
            "frame_id_required",
            "Native Jetson preprocessing requires a positive frame_id."
        };
    }

    long long capture_ts_ns = 0;
    if (!read_int_field(json, "capture_ts_ns", &capture_ts_ns) || capture_ts_ns <= 0) {
        return {
            false,
            "capture_ts_ns_required",
            "Native Jetson preprocessing requires a positive capture_ts_ns."
        };
    }

    long long dmabuf_fd = -1;
    if (!read_int_field(json, "dmabuf_fd", &dmabuf_fd)) {
        dmabuf_fd = -1;
    }
    long long gst_buffer_ptr = 0;
    if (!read_int_field(json, "gst_buffer_ptr", &gst_buffer_ptr)) {
        gst_buffer_ptr = 0;
    }
    if (dmabuf_fd < 0 && gst_buffer_ptr <= 0) {
        return {
            false,
            "frame_resource_handle_required",
            "Native Jetson preprocessing requires a valid dmabuf_fd or gst_buffer_ptr."
        };
    }

    std::string resource_kind;
    if (!read_string_field(json, "resource_kind", &resource_kind)
        || resource_kind != "gstreamer_sample") {
        return {
            false,
            "unsupported_resource_kind",
            "Expected resource_kind=gstreamer_sample."
        };
    }

    std::string resource_memory;
    if (!read_string_field(json, "resource_memory", &resource_memory)
        || (resource_memory != "dmabuf" && resource_memory != "nvmm")) {
        return {
            false,
            "unsupported_resource_memory",
            "Expected resource_memory=dmabuf or resource_memory=nvmm."
        };
    }

    std::string resource_source;
    if (!read_string_field(json, "resource_source", &resource_source)
        || (resource_source != "appsink" && resource_source != "deepstream_pad")) {
        return {
            false,
            "unsupported_resource_source",
            "Expected resource_source=appsink or deepstream_pad."
        };
    }

    std::string resource_pixel_format;
    if (!read_string_field(json, "resource_pixel_format", &resource_pixel_format)
        || resource_pixel_format != "NV12") {
        return {
            false,
            "unsupported_resource_pixel_format",
            "Expected resource_pixel_format=NV12."
        };
    }

    std::string pixel_format;
    if (!read_string_field(json, "pixel_format", &pixel_format) || pixel_format != "NV12") {
        return {false, "unsupported_pixel_format", "Expected pixel_format=NV12."};
    }

    std::string dtype;
    if (!read_string_field(json, "dtype", &dtype)
        || (dtype != "float32" && dtype != "float16")) {
        return {false, "unsupported_dtype", "Expected dtype=float32 or dtype=float16."};
    }

    long long width = 0;
    long long height = 0;
    for (const char* key : {"width", "height", "source_width", "source_height"}) {
        long long value = 0;
        if (!read_int_field(json, key, &value) || value <= 0) {
            return {
                false,
                "invalid_geometry",
                std::string("Expected positive integer field: ") + key + "."
            };
        }
        if (std::strcmp(key, "width") == 0) {
            width = value;
        } else if (std::strcmp(key, "height") == 0) {
            height = value;
        }
    }
    long long resource_width = 0;
    if (!read_int_field(json, "resource_width", &resource_width) || resource_width <= 0) {
        return {
            false,
            "invalid_resource_geometry",
            "Expected positive integer field: resource_width."
        };
    }
    long long resource_height = 0;
    if (!read_int_field(json, "resource_height", &resource_height) || resource_height <= 0) {
        return {
            false,
            "invalid_resource_geometry",
            "Expected positive integer field: resource_height."
        };
    }
    if (resource_width != width || resource_height != height) {
        return {
            false,
            "resource_geometry_mismatch",
            "Expected resource_width/resource_height to match width/height."
        };
    }
    if (resource_pixel_format != pixel_format) {
        return {
            false,
            "resource_format_mismatch",
            "Expected resource_pixel_format to match pixel_format."
        };
    }

    for (const char* key : {"roi_offset_x", "roi_offset_y"}) {
        long long value = 0;
        if (!read_int_field(json, key, &value) || value < 0) {
            return {
                false,
                "invalid_roi_offset",
                std::string("Expected non-negative integer field: ") + key + "."
            };
        }
    }

    bool needs_resize = false;
    if (!read_bool_field(json, "needs_resize", &needs_resize)) {
        return {false, "needs_resize_required", "Expected boolean field: needs_resize."};
    }

    std::string model_shape_json;
    if (!read_object_field(json, "model_shape", &model_shape_json)) {
        return {
            false,
            "model_shape_required",
            "Expected model_shape object with batch/channels/height/width."
        };
    }
    std::vector<long long> model_shape;
    for (const char* key : {"batch", "channels", "height", "width"}) {
        long long value = 0;
        if (!read_int_field(model_shape_json, key, &value) || value <= 0) {
            return {
                false,
                "invalid_model_shape",
                std::string("Expected positive integer field: model_shape.") + key + "."
            };
        }
        model_shape.push_back(value);
    }

    std::vector<long long> nchw;
    if (!read_int_array_field(json, "nchw", &nchw) || nchw.size() != 4) {
        return {false, "invalid_nchw", "Expected nchw=[N,C,H,W]."};
    }
    if (nchw[0] <= 0 || nchw[1] <= 0 || nchw[2] <= 0 || nchw[3] <= 0) {
        return {false, "invalid_nchw", "NCHW dimensions must be positive."};
    }
    if (nchw[1] != 1 && nchw[1] != 3 && nchw[1] != 4) {
        return {false, "unsupported_channels", "NCHW channel count must be 1, 3, or 4."};
    }
    if (model_shape != nchw) {
        return {
            false,
            "model_shape_mismatch",
            "Expected model_shape to match nchw exactly."
        };
    }
    const bool expected_resize = width != nchw[3] || height != nchw[2];
    if (needs_resize != expected_resize) {
        return {
            false,
            "needs_resize_mismatch",
            "Expected needs_resize to match frame and model geometry."
        };
    }

    return {true, "", ""};
}

}  // namespace

int status_json_impl(char* status_json, size_t status_json_size) {
    write_json(
        status_json,
        status_json_size,
        "{\"available\":false,"
        "\"ready\":false,"
        "\"backend\":\"novasight_jetson_preprocess_native:reference\","
        "\"reason\":\"jetson_cuda_preprocess_not_compiled\","
        "\"detail\":\"Reference ABI library is buildable but does not convert "
        "DMABUF/NvBufSurface/EGL/CUDA resources. Replace this implementation "
        "with Jetson GPU preprocessing before enabling nvmm inference.\","
        "\"capabilities\":{"
        "\"memory\":[\"dmabuf\",\"nvmm\"],"
        "\"resource_kind\":[\"gstreamer_sample\"],"
        "\"resource_source\":[\"appsink\",\"deepstream_pad\"],"
        "\"formats\":[\"NV12\"],"
        "\"dtypes\":[\"float32\",\"float16\"]},"
        "\"abi_version\":1}");
    return 0;
}

int prepare_tensor_impl(
    const char* payload_json,
    char* result_json,
    size_t result_json_size
) {
    const PayloadValidation validation = validate_payload(payload_json);
    if (!validation.valid) {
        write_error_json(
            result_json,
            result_json_size,
            validation.reason,
            validation.detail);
        return 2;
    }
    write_json(
        result_json,
        result_json_size,
        "{\"reason\":\"jetson_cuda_preprocess_not_compiled\","
        "\"validated\":true,"
        "\"detail\":\"Reference ABI library validated the payload but does not produce a DeviceTensor. "
        "Implement DMABUF/NvBufSurface/EGL/CUDA to NCHW TensorRT input "
        "conversion and return device_ptr/nbytes.\"}");
    return 3;
}

extern "C" uint32_t novasight_abi_version(void) {
    return NOVASIGHT_JETSON_PREPROCESS_ABI_VERSION;
}

extern "C" int novasight_status_json(char* status_json, size_t status_json_size) {
    try {
        return status_json_impl(status_json, status_json_size);
    } catch (...) {
        write_json(
            status_json,
            status_json_size,
            "{\"available\":false,\"ready\":false,\"reason\":\"native_exception\"}"
        );
        return 2;
    }
}

extern "C" int novasight_prepare_tensor_json(
    const char* payload_json,
    char* result_json,
    size_t result_json_size
) {
    try {
        return prepare_tensor_impl(payload_json, result_json, result_json_size);
    } catch (...) {
        write_json(
            result_json,
            result_json_size,
            "{\"reason\":\"native_exception\",\"detail\":\"unexpected reference ABI failure\"}"
        );
        return 2;
    }
}

extern "C" int novasight_release_tensor(uint64_t release_token) {
    (void)release_token;
    return 0;
}
