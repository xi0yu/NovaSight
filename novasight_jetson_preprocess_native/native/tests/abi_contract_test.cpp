#include "novasight_jetson_preprocess_native.h"

#include <cassert>
#include <cstring>
#include <string>

namespace {

constexpr const char* kValidDeepStreamPayload = R"json({
  "frame_id": 4294967297,
  "capture_ts_ns": 123456789,
  "dmabuf_fd": -1,
  "gst_buffer_ptr": 4096,
  "resource_kind": "gstreamer_sample",
  "resource_memory": "nvmm",
  "resource_source": "deepstream_pad",
  "resource_pixel_format": "NV12",
  "resource_width": 640,
  "resource_height": 640,
  "pixel_format": "NV12",
  "width": 640,
  "height": 640,
  "source_width": 640,
  "source_height": 640,
  "roi_offset_x": 0,
  "roi_offset_y": 0,
  "needs_resize": false,
  "model_shape": {"batch": 1, "channels": 3, "height": 640, "width": 640},
  "nchw": [1, 3, 640, 640],
  "dtype": "float16"
})json";

}  // namespace

int main() {
    assert(novasight_abi_version() == NOVASIGHT_JETSON_PREPROCESS_ABI_VERSION);

    char status[4096] = {0};
    assert(novasight_status_json(status, sizeof(status)) == 0);
    assert(std::strstr(status, "\"abi_version\":1") != nullptr);
    assert(std::strstr(status, "deepstream_pad") != nullptr);

    char result[4096] = {0};
    const int result_code = novasight_prepare_tensor_json(
        kValidDeepStreamPayload,
        result,
        sizeof(result)
    );
    // The portable reference validates the complete request, then fails
    // closed because it has no CUDA implementation. A schema rejection is 2.
    assert(result_code == 3);
    assert(std::strstr(result, "jetson_cuda_preprocess_not_compiled") != nullptr);

    std::string invalid_source(kValidDeepStreamPayload);
    const std::string needle = "deepstream_pad";
    invalid_source.replace(invalid_source.find(needle), needle.size(), "unknown_source");
    std::memset(result, 0, sizeof(result));
    assert(novasight_prepare_tensor_json(
               invalid_source.c_str(),
               result,
               sizeof(result)
           ) == 2);
    assert(std::strstr(result, "unsupported_resource_source") != nullptr);

    std::string mismatched_shape(kValidDeepStreamPayload);
    const std::string shape_needle = "\"nchw\": [1, 3, 640, 640]";
    mismatched_shape.replace(
        mismatched_shape.find(shape_needle),
        shape_needle.size(),
        "\"nchw\": [1, 3, 320, 320]"
    );
    std::memset(result, 0, sizeof(result));
    assert(novasight_prepare_tensor_json(
               mismatched_shape.c_str(),
               result,
               sizeof(result)
           ) == 2);
    assert(std::strstr(result, "model_shape_mismatch") != nullptr);

    std::string overflowing_frame_id(kValidDeepStreamPayload);
    const std::string frame_id = "4294967297";
    overflowing_frame_id.replace(
        overflowing_frame_id.find(frame_id),
        frame_id.size(),
        "18446744073709551616"
    );
    std::memset(result, 0, sizeof(result));
    assert(novasight_prepare_tensor_json(
               overflowing_frame_id.c_str(),
               result,
               sizeof(result)
           ) == 2);
    assert(std::strstr(result, "frame_id_required") != nullptr);
    return 0;
}
