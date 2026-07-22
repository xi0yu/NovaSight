#include "novasight_jetson_preprocess_native.h"

#include <cstring>

namespace {

void write_json(char* output, size_t output_size, const char* json) {
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

}  // namespace

extern "C" uint32_t novasight_abi_version(void) {
    return NOVASIGHT_JETSON_PREPROCESS_ABI_VERSION;
}

extern "C" int novasight_status_json(char* status_json, size_t status_json_size) {
    write_json(
        status_json,
        status_json_size,
        "{\"available\":false,"
        "\"ready\":false,"
        "\"backend\":\"novasight_jetson_preprocess_native:jetson_scaffold\","
        "\"reason\":\"jetson_cuda_preprocess_scaffold_only\","
        "\"detail\":\"This scaffold exports the ABI for build-system validation "
        "only. Configure NOVASIGHT_JETSON_PREPROCESS_IMPL=jetson with a real "
        "NOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE before using "
        "capture.memory=nvmm for inference.\","
        "\"capabilities\":{"
        "\"memory\":[\"dmabuf\",\"nvmm\"],"
        "\"resource_kind\":[\"gstreamer_sample\"],"
        "\"resource_source\":[\"appsink\",\"deepstream_pad\"],"
        "\"formats\":[\"NV12\"],"
        "\"dtypes\":[\"float32\",\"float16\"]},"
        "\"abi_version\":1}");
    return 0;
}

extern "C" int novasight_prepare_tensor_json(
    const char* payload_json,
    char* result_json,
    size_t result_json_size
) {
    (void)payload_json;
    write_json(
        result_json,
        result_json_size,
        "{\"reason\":\"jetson_cuda_preprocess_scaffold_only\","
        "\"detail\":\"No DeviceTensor was created. Use a real Jetson "
        "CUDA/NvBufSurface production source.\"}");
    return 3;
}

extern "C" int novasight_release_tensor(uint64_t release_token) {
    (void)release_token;
    return 0;
}
