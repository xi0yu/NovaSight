#include "novasight_tensorrt_runtime.h"

#include <cassert>
#include <cstring>

int main() {
    static_assert(sizeof(novasight_tensor_spec) == 216);
    static_assert(sizeof(novasight_engine_spec) == 1952);
    static_assert(sizeof(novasight_device_tensor_view) == 96);
    static_assert(sizeof(novasight_host_tensor_view) == 232);
    assert(novasight_tensorrt_abi_version() == NOVASIGHT_TENSORRT_RUNTIME_ABI_VERSION);
    const uint64_t shape[] = {1, 3, 640, 640};
    novasight_tensorrt_engine* engine = reinterpret_cast<novasight_tensorrt_engine*>(1);
    novasight_engine_spec spec{};
    char error[256] = {};
    assert(novasight_tensorrt_create(
               "/tmp/model.engine",
               shape,
               4,
               &engine,
               &spec,
               error,
               sizeof(error)
           ) == 2);
    assert(engine == nullptr);
    assert(std::strstr(error, "fail-closed") != nullptr);

    uint32_t output_count = 99;
    std::memset(error, 0, sizeof(error));
    assert(novasight_tensorrt_execute(
               nullptr,
               nullptr,
               nullptr,
               0,
               &output_count,
               error,
               sizeof(error)
           ) == 2);
    assert(output_count == 0);
    assert(std::strstr(error, "cannot execute") != nullptr);
    return 0;
}
