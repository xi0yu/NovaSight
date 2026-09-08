#include "novasight_tensorrt_runtime.h"

#include <cstring>

namespace {

void write_error(char* output, size_t output_size, const char* message) noexcept {
    if (output == nullptr || output_size == 0) return;
    if (message == nullptr) message = "";
    const size_t length = std::strlen(message);
    const size_t copied = length < output_size - 1 ? length : output_size - 1;
    std::memcpy(output, message, copied);
    output[copied] = '\0';
}

}  // namespace

extern "C" uint32_t novasight_tensorrt_abi_version(void) {
    return NOVASIGHT_TENSORRT_RUNTIME_ABI_VERSION;
}

extern "C" int novasight_tensorrt_environment_query(
    novasight_tensorrt_environment* environment_out,
    char* error_out,
    size_t error_out_size
) {
    if (environment_out != nullptr) {
        std::memset(environment_out, 0, sizeof(*environment_out));
    }
    write_error(
        error_out,
        error_out_size,
        "reference TensorRT runtime has no CUDA environment"
    );
    return 2;
}

extern "C" int novasight_tensorrt_create(
    const char* engine_path,
    const uint64_t* requested_input_shape,
    uint32_t requested_input_rank,
    novasight_tensorrt_engine** engine_out,
    novasight_engine_spec* spec_out,
    char* error_out,
    size_t error_out_size
) {
    (void)engine_path;
    (void)requested_input_shape;
    (void)requested_input_rank;
    if (engine_out != nullptr) *engine_out = nullptr;
    if (spec_out != nullptr) std::memset(spec_out, 0, sizeof(*spec_out));
    write_error(
        error_out,
        error_out_size,
        "reference TensorRT runtime is fail-closed; build the production Jetson implementation"
    );
    return 2;
}

extern "C" int novasight_tensorrt_execute(
    novasight_tensorrt_engine* engine,
    const novasight_device_tensor_view* input,
    novasight_host_tensor_view* outputs,
    uint32_t output_capacity,
    uint32_t* output_count,
    char* error_out,
    size_t error_out_size
) {
    (void)engine;
    (void)input;
    (void)outputs;
    (void)output_capacity;
    if (output_count != nullptr) *output_count = 0;
    write_error(error_out, error_out_size, "reference TensorRT runtime cannot execute");
    return 2;
}

extern "C" int novasight_tensorrt_enqueue_device(
    novasight_tensorrt_engine* engine, const novasight_device_tensor_view* input, uint64_t input_ready_event,
    novasight_device_tensor_view* outputs, uint32_t output_capacity,
    uint32_t* output_count, uint64_t* stream_out, char* error_out, size_t error_out_size
) {
    (void)engine; (void)input; (void)input_ready_event; (void)outputs; (void)output_capacity;
    if (output_count != nullptr) *output_count = 0;
    if (stream_out != nullptr) *stream_out = 0;
    write_error(error_out, error_out_size, "reference TensorRT runtime cannot enqueue CUDA work");
    return 2;
}

extern "C" int novasight_tensorrt_finish_device(
    novasight_tensorrt_engine* engine, char* error_out, size_t error_out_size
) {
    (void)engine;
    write_error(error_out, error_out_size, "reference TensorRT runtime has no CUDA work to finish");
    return 2;
}

extern "C" int novasight_tensorrt_probe_zero(
    novasight_tensorrt_engine* engine,
    novasight_host_tensor_view* outputs,
    uint32_t output_capacity,
    uint32_t* output_count,
    char* error_out,
    size_t error_out_size
) {
    (void)engine;
    (void)outputs;
    (void)output_capacity;
    if (output_count != nullptr) *output_count = 0;
    write_error(error_out, error_out_size, "reference TensorRT runtime cannot probe");
    return 2;
}

extern "C" void novasight_tensorrt_destroy(novasight_tensorrt_engine* engine) {
    (void)engine;
}
