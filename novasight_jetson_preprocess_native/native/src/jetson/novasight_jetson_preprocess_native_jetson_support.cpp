#include "novasight_jetson_preprocess_native_jetson_support.h"

#include "novasight_jetson_preprocess_native.h"

#include <cctype>
#include <cstdlib>
#include <cstring>

namespace novasight::jetson_preprocess {
namespace {

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

bool read_int_field(const std::string& json, const char* key, int* value) {
    const char* cursor = field_value_start(json, key);
    if (cursor == nullptr || std::strncmp(cursor, "null", 4) == 0) {
        return false;
    }
    char* end = nullptr;
    const long parsed = std::strtol(cursor, &end, 10);
    if (end == cursor || !json_value_terminator(*end)) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

bool read_uint64_field(const std::string& json, const char* key, uint64_t* value) {
    const char* cursor = field_value_start(json, key);
    if (cursor == nullptr || std::strncmp(cursor, "null", 4) == 0) {
        return false;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(cursor, &end, 10);
    if (end == cursor || !json_value_terminator(*end)) {
        return false;
    }
    *value = static_cast<uint64_t>(parsed);
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
    std::vector<int>* values
) {
    const char* cursor = field_value_start(json, key);
    if (cursor == nullptr || *cursor != '[') {
        return false;
    }
    ++cursor;
    std::vector<int> parsed_values;
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
        const long parsed = std::strtol(cursor, &end, 10);
        if (end == cursor || !json_value_terminator(*end)) {
            return false;
        }
        parsed_values.push_back(static_cast<int>(parsed));
        cursor = end;
        expect_value = false;
    }
    if (*cursor != ']' || !json_value_terminator(cursor[1])) {
        return false;
    }
    *values = parsed_values;
    return true;
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\b':
                escaped += "\\b";
                break;
            case '\f':
                escaped += "\\f";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (ch < 0x20) {
                    const char digits[] = "0123456789abcdef";
                    escaped += "\\u00";
                    escaped.push_back(digits[(ch >> 4) & 0x0F]);
                    escaped.push_back(digits[ch & 0x0F]);
                } else {
                    escaped.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    return escaped;
}

RequestValidation invalid(
    const std::string& reason,
    const std::string& detail,
    const TensorRequest& request = TensorRequest()
) {
    return {false, reason, detail, request};
}

}  // namespace

void write_json(char* output, size_t output_size, const std::string& json) {
    if (output == nullptr || output_size == 0) {
        return;
    }
    const size_t length = json.size();
    const size_t copy_length = length < output_size - 1 ? length : output_size - 1;
    std::memcpy(output, json.c_str(), copy_length);
    output[copy_length] = '\0';
}

void write_error_json(
    char* output,
    size_t output_size,
    const std::string& reason,
    const std::string& detail
) {
    write_json(
        output,
        output_size,
        "{\"reason\":\"" + json_escape(reason) + "\","
        "\"detail\":\"" + json_escape(detail) + "\"}"
    );
}

int dtype_size_bytes(const std::string& dtype) {
    if (dtype == "float32") {
        return 4;
    }
    if (dtype == "float16") {
        return 2;
    }
    return 0;
}

uint64_t tensor_nbytes(const TensorRequest& request) {
    if (request.nchw.size() != 4) {
        return 0;
    }
    const int dtype_size = dtype_size_bytes(request.dtype);
    if (dtype_size <= 0) {
        return 0;
    }
    uint64_t total = static_cast<uint64_t>(dtype_size);
    for (const int dim : request.nchw) {
        if (dim <= 0) {
            return 0;
        }
        total *= static_cast<uint64_t>(dim);
    }
    return total;
}

RequestValidation parse_tensor_request(const char* payload_json) {
    if (payload_json == nullptr || payload_json[0] == '\0') {
        return invalid("payload_required", "payload_json is empty");
    }
    const std::string json(payload_json);
    TensorRequest request;
    if (!read_int_field(json, "frame_id", &request.frame_id) || request.frame_id <= 0) {
        return invalid(
            "frame_id_required",
            "Native Jetson preprocessing requires a positive frame_id.",
            request
        );
    }
    if (!read_uint64_field(json, "capture_ts_ns", &request.capture_ts_ns)
        || request.capture_ts_ns == 0) {
        return invalid(
            "capture_ts_ns_required",
            "Native Jetson preprocessing requires a positive capture_ts_ns.",
            request
        );
    }
    if (!read_int_field(json, "dmabuf_fd", &request.dmabuf_fd) || request.dmabuf_fd < 0) {
        return invalid(
            "dmabuf_fd_required",
            "Native Jetson preprocessing requires a valid dmabuf_fd.",
            request
        );
    }
    if (!read_string_field(json, "resource_kind", &request.resource_kind)
        || request.resource_kind != "gstreamer_sample") {
        return invalid("unsupported_resource_kind", "Expected resource_kind=gstreamer_sample.", request);
    }
    if (!read_string_field(json, "resource_memory", &request.resource_memory)
        || (request.resource_memory != "dmabuf" && request.resource_memory != "nvmm")) {
        return invalid(
            "unsupported_resource_memory",
            "Expected resource_memory=dmabuf or resource_memory=nvmm.",
            request
        );
    }
    if (!read_string_field(json, "resource_source", &request.resource_source)
        || request.resource_source != "appsink") {
        return invalid("unsupported_resource_source", "Expected resource_source=appsink.", request);
    }
    if (!read_string_field(json, "resource_pixel_format", &request.resource_pixel_format)
        || request.resource_pixel_format != "NV12") {
        return invalid(
            "unsupported_resource_pixel_format",
            "Expected resource_pixel_format=NV12.",
            request
        );
    }
    if (!read_string_field(json, "pixel_format", &request.pixel_format)
        || request.pixel_format != "NV12") {
        return invalid("unsupported_pixel_format", "Expected pixel_format=NV12.", request);
    }
    if (!read_string_field(json, "dtype", &request.dtype) || dtype_size_bytes(request.dtype) <= 0) {
        return invalid("unsupported_dtype", "Expected dtype=float32 or dtype=float16.", request);
    }
    for (const char* key : {"width", "height", "source_width", "source_height"}) {
        int* target = nullptr;
        if (std::strcmp(key, "width") == 0) {
            target = &request.width;
        } else if (std::strcmp(key, "height") == 0) {
            target = &request.height;
        } else if (std::strcmp(key, "source_width") == 0) {
            target = &request.source_width;
        } else {
            target = &request.source_height;
        }
        if (!read_int_field(json, key, target) || *target <= 0) {
            return invalid("invalid_geometry", std::string("Expected positive integer field: ") + key + ".", request);
        }
    }
    if (!read_int_field(json, "resource_width", &request.resource_width)
        || request.resource_width <= 0) {
        return invalid(
            "invalid_resource_geometry",
            "Expected positive integer field: resource_width.",
            request
        );
    }
    if (!read_int_field(json, "resource_height", &request.resource_height)
        || request.resource_height <= 0) {
        return invalid(
            "invalid_resource_geometry",
            "Expected positive integer field: resource_height.",
            request
        );
    }
    if (request.resource_width != request.width || request.resource_height != request.height) {
        return invalid(
            "resource_geometry_mismatch",
            "Expected resource_width/resource_height to match width/height.",
            request
        );
    }
    if (request.resource_pixel_format != request.pixel_format) {
        return invalid(
            "resource_format_mismatch",
            "Expected resource_pixel_format to match pixel_format.",
            request
        );
    }
    if (!read_int_field(json, "roi_offset_x", &request.roi_offset_x)
        || request.roi_offset_x < 0) {
        return invalid(
            "invalid_roi_offset",
            "Expected non-negative integer field: roi_offset_x.",
            request
        );
    }
    if (!read_int_field(json, "roi_offset_y", &request.roi_offset_y)
        || request.roi_offset_y < 0) {
        return invalid(
            "invalid_roi_offset",
            "Expected non-negative integer field: roi_offset_y.",
            request
        );
    }
    if (!read_bool_field(json, "needs_resize", &request.needs_resize)) {
        return invalid(
            "needs_resize_required",
            "Expected boolean field: needs_resize.",
            request
        );
    }
    std::string model_shape_json;
    if (!read_object_field(json, "model_shape", &model_shape_json)) {
        return invalid(
            "model_shape_required",
            "Expected model_shape object with batch/channels/height/width.",
            request
        );
    }
    for (const char* key : {"batch", "channels", "height", "width"}) {
        int value = 0;
        if (!read_int_field(model_shape_json, key, &value) || value <= 0) {
            return invalid(
                "invalid_model_shape",
                std::string("Expected positive integer field: model_shape.") + key + ".",
                request
            );
        }
    }
    if (!read_int_array_field(json, "nchw", &request.nchw) || request.nchw.size() != 4) {
        return invalid("invalid_nchw", "Expected nchw=[N,C,H,W].", request);
    }
    for (const int dim : request.nchw) {
        if (dim <= 0) {
            return invalid("invalid_nchw", "NCHW dimensions must be positive.", request);
        }
    }
    if (request.nchw[1] != 1 && request.nchw[1] != 3 && request.nchw[1] != 4) {
        return invalid("unsupported_channels", "NCHW channel count must be 1, 3, or 4.", request);
    }
    return {true, "", "", request};
}

std::string unavailable_status_json(
    const std::string& backend,
    const std::string& reason,
    const std::string& detail
) {
    return "{\"available\":false,"
           "\"ready\":false,"
           "\"backend\":\"" + json_escape(backend) + "\","
           "\"reason\":\"" + json_escape(reason) + "\","
           "\"detail\":\"" + json_escape(detail) + "\","
           "\"capabilities\":{"
           "\"memory\":[\"dmabuf\",\"nvmm\"],"
           "\"resource_kind\":[\"gstreamer_sample\"],"
           "\"resource_source\":[\"appsink\"],"
           "\"formats\":[\"NV12\"],"
           "\"dtypes\":[\"float32\",\"float16\"]},"
           "\"abi_version\":" + std::to_string(NOVASIGHT_JETSON_PREPROCESS_ABI_VERSION) + "}";
}

}  // namespace novasight::jetson_preprocess
