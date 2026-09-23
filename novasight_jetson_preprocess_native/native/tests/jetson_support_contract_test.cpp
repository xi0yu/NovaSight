#include "novasight_jetson_preprocess_native_jetson_support.h"
#include "novasight_nvbufsurface_payload.h"

#include <cassert>
#include <cstring>
#include <string>

namespace {

constexpr const char* kPayload = R"json({
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
    struct FakeSurface {
        unsigned long long first_word = 0xDEADBEEF;
        unsigned long long second_word = 0;
    } fake_surface;
    void* mapped = novasight::jetson_preprocess::checked_nvbufsurface_payload(
        &fake_surface,
        sizeof(fake_surface),
        sizeof(fake_surface)
    );
    assert(mapped == &fake_surface);
    assert(mapped != reinterpret_cast<void*>(fake_surface.first_word));

    const auto valid = novasight::jetson_preprocess::parse_tensor_request(kPayload);
    assert(valid.valid);
    assert(valid.request.frame_id == 4294967297ULL);
    assert(valid.request.resource_source == "deepstream_pad");
    assert(valid.request.nchw.size() == 4);

    std::string mismatched_resize(kPayload);
    const std::string needle = "\"needs_resize\": false";
    mismatched_resize.replace(
        mismatched_resize.find(needle),
        needle.size(),
        "\"needs_resize\": true"
    );
    const auto invalid = novasight::jetson_preprocess::parse_tensor_request(
        mismatched_resize.c_str()
    );
    assert(!invalid.valid);
    assert(invalid.reason == "needs_resize_mismatch");

    char exact[5] = {};
    assert(novasight::jetson_preprocess::write_json(exact, sizeof(exact), "test"));
    assert(std::strcmp(exact, "test") == 0);
    char too_small[4] = {'x', 'x', 'x', 'x'};
    assert(!novasight::jetson_preprocess::write_json(
        too_small,
        sizeof(too_small),
        "test"
    ));
    assert(too_small[0] == '\0');

    std::string overflowing_frame_id(kPayload);
    const std::string frame_id = "4294967297";
    overflowing_frame_id.replace(
        overflowing_frame_id.find(frame_id),
        frame_id.size(),
        "18446744073709551616"
    );
    const auto overflow = novasight::jetson_preprocess::parse_tensor_request(
        overflowing_frame_id.c_str()
    );
    assert(!overflow.valid);
    assert(overflow.reason == "frame_id_required");
    return 0;
}
