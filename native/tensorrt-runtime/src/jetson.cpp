#include "novasight_tensorrt_runtime.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <climits>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void write_error(char* output, size_t output_size, const char* message) noexcept {
    if (output == nullptr || output_size == 0) return;
    if (message == nullptr) message = "";
    const size_t length = std::strlen(message);
    const size_t copied = length < output_size - 1 ? length : output_size - 1;
    std::memcpy(output, message, copied);
    output[copied] = '\0';
}

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        (void)message;
        if (severity > Severity::kWARNING) return;
    }
};

template <typename T>
struct TrtDelete {
    void operator()(T* value) const noexcept { delete value; }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

uint32_t dtype_code(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return NOVASIGHT_TENSOR_DTYPE_FLOAT32;
        case nvinfer1::DataType::kHALF:
            return NOVASIGHT_TENSOR_DTYPE_FLOAT16;
        default:
            throw std::runtime_error("TensorRT tensor dtype must be float32 or float16");
    }
}

uint64_t dtype_bytes(uint32_t dtype) {
    if (dtype == NOVASIGHT_TENSOR_DTYPE_FLOAT32) return 4;
    if (dtype == NOVASIGHT_TENSOR_DTYPE_FLOAT16) return 2;
    throw std::runtime_error("unsupported NovaSight tensor dtype");
}

uint64_t checked_nbytes(const nvinfer1::Dims& dims, uint32_t dtype) {
    if (dims.nbDims <= 0 || dims.nbDims > static_cast<int>(NOVASIGHT_TENSORRT_MAX_RANK)) {
        throw std::runtime_error("TensorRT tensor rank is unsupported");
    }
    uint64_t elements = 1;
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) {
            throw std::runtime_error("TensorRT tensor shape is unresolved");
        }
        const uint64_t dimension = static_cast<uint64_t>(dims.d[index]);
        if (elements > std::numeric_limits<uint64_t>::max() / dimension) {
            throw std::runtime_error("TensorRT tensor element count overflow");
        }
        elements *= dimension;
    }
    const uint64_t width = dtype_bytes(dtype);
    if (elements > std::numeric_limits<uint64_t>::max() / width) {
        throw std::runtime_error("TensorRT tensor byte size overflow");
    }
    return elements * width;
}

novasight_tensor_spec tensor_spec(
    const char* name,
    const nvinfer1::Dims& dims,
    nvinfer1::DataType dtype
) {
    if (name == nullptr || name[0] == '\0') {
        throw std::runtime_error("TensorRT tensor name is empty");
    }
    if (std::strlen(name) >= NOVASIGHT_TENSORRT_MAX_TENSOR_NAME) {
        throw std::runtime_error("TensorRT tensor name exceeds ABI capacity");
    }
    novasight_tensor_spec spec{};
    std::memcpy(spec.name, name, std::strlen(name));
    spec.rank = static_cast<uint32_t>(dims.nbDims);
    for (int index = 0; index < dims.nbDims; ++index) {
        spec.dimensions[index] = dims.d[index];
    }
    spec.dtype = dtype_code(dtype);
    spec.nbytes = checked_nbytes(dims, spec.dtype);
    return spec;
}

void validate_tensor_storage(const nvinfer1::ICudaEngine& engine, const char* name) {
    if (engine.getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE) {
        throw std::runtime_error("NovaSight TensorRT tensors must use device memory");
    }
    if (engine.isShapeInferenceIO(name)) {
        throw std::runtime_error("NovaSight does not accept TensorRT shape I/O tensors");
    }
    if (engine.getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR
        || engine.getTensorVectorizedDim(name) != -1) {
        throw std::runtime_error("NovaSight requires linear non-vectorized TensorRT tensors");
    }
}

void cuda_check(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return;
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorName(result)
        + ": " + cudaGetErrorString(result)
    );
}

std::vector<char> read_engine(const char* path) {
    if (path == nullptr || path[0] == '\0') {
        throw std::runtime_error("TensorRT engine path is empty");
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("failed to open TensorRT engine");
    const std::streamoff length = input.tellg();
    if (length <= 0) throw std::runtime_error("TensorRT engine file is empty");
    if (static_cast<uint64_t>(length) > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("TensorRT engine file is too large");
    }
    std::vector<char> bytes(static_cast<size_t>(length));
    input.seekg(0, std::ios::beg);
    if (!input.read(bytes.data(), length)) {
        throw std::runtime_error("failed to read complete TensorRT engine");
    }
    return bytes;
}

struct OutputSlot {
    novasight_tensor_spec spec{};
    void* device = nullptr;
    void* host = nullptr;

    OutputSlot() = default;
    OutputSlot(const OutputSlot&) = delete;
    OutputSlot& operator=(const OutputSlot&) = delete;

    OutputSlot(OutputSlot&& other) noexcept
        : spec(other.spec), device(other.device), host(other.host) {
        other.device = nullptr;
        other.host = nullptr;
    }

    ~OutputSlot() noexcept {
        if (device != nullptr) cudaFree(device);
        if (host != nullptr) cudaFreeHost(host);
    }
};

struct StreamDrainGuard {
    cudaStream_t stream = nullptr;
    bool armed = false;

    cudaError_t drain() noexcept {
        if (!armed || stream == nullptr) return cudaSuccess;
        armed = false;
        return cudaStreamSynchronize(stream);
    }

    ~StreamDrainGuard() noexcept {
        if (armed) (void)drain();
    }
};

std::string cuda_failure_detail(const char* operation, cudaError_t result) {
    return std::string(operation) + " failed: " + cudaGetErrorName(result)
           + ": " + cudaGetErrorString(result);
}

}  // namespace

struct novasight_tensorrt_engine {
    Logger logger;
    TrtPtr<nvinfer1::IRuntime> runtime;
    TrtPtr<nvinfer1::ICudaEngine> engine;
    TrtPtr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    novasight_engine_spec spec{};
    std::vector<OutputSlot> outputs;
    bool device_pending = false;
    bool device_failed = false;
    uint64_t bound_input = 0;
    bool warmed_up = false;
    cudaGraphExec_t device_graph = nullptr;

    void discard_graph() noexcept {
        if (device_graph) cudaGraphExecDestroy(device_graph);
        device_graph = nullptr;
    }

    ~novasight_tensorrt_engine() noexcept {
        if (device_pending && stream != nullptr) (void)cudaStreamSynchronize(stream);
        discard_graph();
        if (stream != nullptr) cudaStreamDestroy(stream);
    }
};

namespace {

std::unique_ptr<novasight_tensorrt_engine> create_engine(
    const char* engine_path,
    const uint64_t* requested_input_shape,
    uint32_t requested_input_rank
) {
    const bool auto_shape = requested_input_shape == nullptr && requested_input_rank == 0;
    if (!auto_shape && (requested_input_shape == nullptr || requested_input_rank != 4)) {
        throw std::runtime_error("NovaSight requires an automatic or requested NCHW rank-4 input shape");
    }
    auto owner = std::make_unique<novasight_tensorrt_engine>();
    const std::vector<char> bytes = read_engine(engine_path);
    owner->runtime.reset(nvinfer1::createInferRuntime(owner->logger));
    if (!owner->runtime) throw std::runtime_error("createInferRuntime returned null");
    owner->engine.reset(owner->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!owner->engine) throw std::runtime_error("deserializeCudaEngine returned null");
    owner->context.reset(owner->engine->createExecutionContext());
    if (!owner->context) throw std::runtime_error("createExecutionContext returned null");

    const int io_count = owner->engine->getNbIOTensors();
    const char* input_name = nullptr;
    std::vector<const char*> output_names;
    for (int index = 0; index < io_count; ++index) {
        const char* name = owner->engine->getIOTensorName(index);
        if (name == nullptr || name[0] == '\0') {
            throw std::runtime_error("TensorRT engine exposed an unnamed I/O tensor");
        }
        const auto mode = owner->engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (input_name != nullptr) {
                throw std::runtime_error("NovaSight requires exactly one TensorRT input");
            }
            input_name = name;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_names.push_back(name);
        }
    }
    if (input_name == nullptr || output_names.empty()) {
        throw std::runtime_error("TensorRT engine is missing input or output tensors");
    }
    if (output_names.size() > NOVASIGHT_TENSORRT_MAX_OUTPUTS) {
        throw std::runtime_error("TensorRT output count exceeds ABI capacity");
    }

    const nvinfer1::Dims engine_input = owner->engine->getTensorShape(input_name);
    if (engine_input.nbDims <= 0 || engine_input.nbDims > static_cast<int>(NOVASIGHT_TENSORRT_MAX_RANK)) {
        throw std::runtime_error("TensorRT input rank is unsupported");
    }
    nvinfer1::Dims requested = engine_input;
    if (auto_shape && std::any_of(
            engine_input.d,
            engine_input.d + engine_input.nbDims,
            [](int dimension) { return dimension < 0; })) {
        if (owner->engine->getNbOptimizationProfiles() <= 0) {
            throw std::runtime_error(
                "dynamic TensorRT input has no optimization profile for automatic inspection"
            );
        }
        requested = owner->engine->getProfileShape(
            input_name,
            0,
            nvinfer1::OptProfileSelector::kOPT
        );
    } else if (!auto_shape) {
        requested = {};
        requested.nbDims = static_cast<int>(requested_input_rank);
        for (uint32_t index = 0; index < requested_input_rank; ++index) {
            if (requested_input_shape[index] == 0 || requested_input_shape[index] > INT_MAX) {
                throw std::runtime_error("requested TensorRT input dimension is invalid");
            }
            requested.d[index] = static_cast<int>(requested_input_shape[index]);
        }
    }
    if (engine_input.nbDims != requested.nbDims) {
        throw std::runtime_error("requested input rank does not match TensorRT engine");
    }
    bool dynamic = false;
    for (int index = 0; index < engine_input.nbDims; ++index) {
        if (engine_input.d[index] < 0) {
            dynamic = true;
        } else if (engine_input.d[index] != requested.d[index]) {
            throw std::runtime_error("requested input shape does not match TensorRT engine");
        }
    }
    if (dynamic && !owner->context->setInputShape(input_name, requested)) {
        throw std::runtime_error("TensorRT setInputShape rejected requested dimensions");
    }
    owner->spec.input_dynamic = dynamic ? 1U : 0U;
    owner->spec.selected_profile = 0U;
    if (owner->context->inferShapes(0, nullptr) != 0) {
        throw std::runtime_error("TensorRT input shapes are not fully specified");
    }
    validate_tensor_storage(*owner->engine, input_name);
    const nvinfer1::Dims resolved_input = owner->context->getTensorShape(input_name);
    owner->spec.input = tensor_spec(
        input_name,
        resolved_input,
        owner->engine->getTensorDataType(input_name)
    );
    if (owner->spec.input.rank != 4 || owner->spec.input.dimensions[0] != 1
        || owner->spec.input.dimensions[1] != 3) {
        throw std::runtime_error("NovaSight requires TensorRT input [1,3,H,W]");
    }

    cuda_check(cudaStreamCreate(&owner->stream), "cudaStreamCreate");
    owner->outputs.reserve(output_names.size());
    for (const char* name : output_names) {
        validate_tensor_storage(*owner->engine, name);
        OutputSlot slot;
        slot.spec = tensor_spec(
            name,
            owner->context->getTensorShape(name),
            owner->engine->getTensorDataType(name)
        );
        if (slot.spec.nbytes > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("TensorRT output exceeds host size_t capacity");
        }
        cuda_check(cudaMalloc(&slot.device, static_cast<size_t>(slot.spec.nbytes)), "cudaMalloc output");
        cuda_check(cudaMallocHost(&slot.host, static_cast<size_t>(slot.spec.nbytes)), "cudaMallocHost output");
        if (!owner->context->setTensorAddress(slot.spec.name, slot.device)) {
            throw std::runtime_error("TensorRT rejected output tensor address");
        }
        owner->spec.outputs[owner->outputs.size()] = slot.spec;
        owner->outputs.push_back(std::move(slot));
    }
    owner->spec.output_count = static_cast<uint32_t>(owner->outputs.size());
    return owner;
}

void bind_input(
    novasight_tensorrt_engine* engine,
    const novasight_device_tensor_view* input
) {
    if (engine == nullptr || input == nullptr) {
        throw std::runtime_error("TensorRT execute arguments are null");
    }
    if (engine->device_pending || engine->device_failed) {
        throw std::runtime_error("TensorRT device frame is outstanding or engine has failed");
    }
    const auto& expected = engine->spec.input;
    if (input->device_ptr == 0 || input->nbytes != expected.nbytes
        || input->rank != expected.rank || input->dtype != expected.dtype) {
        throw std::runtime_error("device tensor does not match TensorRT input contract");
    }
    if ((input->device_ptr % 256U) != 0U) {
        throw std::runtime_error("TensorRT device input pointer is not 256-byte aligned");
    }
    for (uint32_t index = 0; index < input->rank; ++index) {
        if (input->dimensions[index] != static_cast<uint64_t>(expected.dimensions[index])) {
            throw std::runtime_error("device tensor shape does not match TensorRT input");
        }
    }
    if (engine->bound_input != input->device_ptr) {
        // Captured contexts cannot be mutated and then replayed with stale bindings.
        engine->discard_graph();
        engine->warmed_up = false;
        if (!engine->context->setTensorAddress(expected.name,
                reinterpret_cast<void*>(static_cast<uintptr_t>(input->device_ptr)))) {
            throw std::runtime_error("TensorRT rejected input tensor address");
        }
        engine->bound_input = input->device_ptr;
    }
}

int execute_engine(
    novasight_tensorrt_engine* engine,
    const novasight_device_tensor_view* input,
    novasight_host_tensor_view* outputs,
    uint32_t output_capacity,
    uint32_t* output_count
) {
    if (engine == nullptr || outputs == nullptr || output_count == nullptr) {
        throw std::runtime_error("TensorRT execute arguments are null");
    }
    *output_count = 0;
    if (output_capacity < engine->outputs.size()) {
        throw std::runtime_error("TensorRT output view capacity is too small");
    }
    bind_input(engine, input);
    engine->discard_graph();
    StreamDrainGuard drain{engine->stream, true};
    if (!engine->context->enqueueV3(engine->stream)) {
        const cudaError_t drain_result = drain.drain();
        if (drain_result != cudaSuccess) {
            throw std::runtime_error(
                "TensorRT enqueueV3 returned false; "
                + cuda_failure_detail("stream drain", drain_result)
            );
        }
        throw std::runtime_error("TensorRT enqueueV3 returned false after stream drain");
    }
    for (const auto& slot : engine->outputs) {
        const cudaError_t copy_result = cudaMemcpyAsync(
            slot.host,
            slot.device,
            static_cast<size_t>(slot.spec.nbytes),
            cudaMemcpyDeviceToHost,
            engine->stream
        );
        if (copy_result != cudaSuccess) {
            const cudaError_t drain_result = drain.drain();
            std::string detail = cuda_failure_detail(
                "cudaMemcpyAsync output D2H",
                copy_result
            );
            if (drain_result != cudaSuccess) {
                detail += "; " + cuda_failure_detail("stream drain", drain_result);
            }
            throw std::runtime_error(detail);
        }
    }
    cuda_check(drain.drain(), "cudaStreamSynchronize");
    engine->warmed_up = true;
    for (size_t index = 0; index < engine->outputs.size(); ++index) {
        outputs[index].host_ptr = engine->outputs[index].host;
        outputs[index].nbytes = engine->outputs[index].spec.nbytes;
        outputs[index].spec = engine->outputs[index].spec;
    }
    *output_count = static_cast<uint32_t>(engine->outputs.size());
    return 0;
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
    try {
        if (environment_out == nullptr) {
            throw std::runtime_error("TensorRT environment output pointer is null");
        }
        int runtime_version = 0;
        int driver_version = 0;
        int device_ordinal = 0;
        cudaDeviceProp properties{};
        cuda_check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
        cuda_check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
        cuda_check(cudaGetDevice(&device_ordinal), "cudaGetDevice");
        cuda_check(
            cudaGetDeviceProperties(&properties, device_ordinal),
            "cudaGetDeviceProperties"
        );
        const size_t device_name_length = std::strlen(properties.name);
        if (device_name_length >= NOVASIGHT_TENSORRT_MAX_DEVICE_NAME) {
            throw std::runtime_error("CUDA device name exceeds ABI capacity");
        }
        environment_out->runtime_abi_version = NOVASIGHT_TENSORRT_RUNTIME_ABI_VERSION;
        environment_out->tensorrt_runtime_version = getInferLibVersion();
        environment_out->cuda_runtime_version = runtime_version;
        environment_out->cuda_driver_version = driver_version;
        environment_out->device_ordinal = device_ordinal;
        environment_out->compute_capability_major = properties.major;
        environment_out->compute_capability_minor = properties.minor;
        environment_out->integrated = properties.integrated != 0 ? 1U : 0U;
        environment_out->total_global_memory =
            static_cast<uint64_t>(properties.totalGlobalMem);
        std::memcpy(environment_out->device_name, properties.name, device_name_length);
        return 0;
    } catch (const std::exception& error) {
        write_error(error_out, error_out_size, error.what());
        return 1;
    } catch (...) {
        write_error(error_out, error_out_size, "unexpected TensorRT environment exception");
        return 2;
    }
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
    if (engine_out != nullptr) *engine_out = nullptr;
    if (spec_out != nullptr) std::memset(spec_out, 0, sizeof(*spec_out));
    try {
        if (engine_out == nullptr || spec_out == nullptr) {
            throw std::runtime_error("TensorRT create output pointers are null");
        }
        auto engine = create_engine(engine_path, requested_input_shape, requested_input_rank);
        *spec_out = engine->spec;
        *engine_out = engine.release();
        return 0;
    } catch (const std::exception& error) {
        write_error(error_out, error_out_size, error.what());
        return 1;
    } catch (...) {
        write_error(error_out, error_out_size, "unexpected TensorRT create exception");
        return 2;
    }
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
    try {
        return execute_engine(engine, input, outputs, output_capacity, output_count);
    } catch (const std::exception& error) {
        if (output_count != nullptr) *output_count = 0;
        write_error(error_out, error_out_size, error.what());
        return 1;
    } catch (...) {
        if (output_count != nullptr) *output_count = 0;
        write_error(error_out, error_out_size, "unexpected TensorRT execute exception");
        return 2;
    }
}

extern "C" int novasight_tensorrt_enqueue_device(
    novasight_tensorrt_engine* engine,
    const novasight_device_tensor_view* input,
    uint64_t input_ready_event,
    novasight_device_tensor_view* outputs,
    uint32_t output_capacity,
    uint32_t* output_count,
    uint64_t* stream_out,
    char* error_out,
    size_t error_out_size
) {
    if (output_count != nullptr) *output_count = 0;
    if (stream_out != nullptr) *stream_out = 0;
    try {
        if (engine == nullptr || outputs == nullptr || output_count == nullptr || stream_out == nullptr
            || output_capacity < engine->outputs.size()) {
            throw std::runtime_error("TensorRT device output arguments/capacity are invalid");
        }
        bind_input(engine, input);
        StreamDrainGuard drain{engine->stream, true};
        if (input_ready_event != 0)
            cuda_check(cudaStreamWaitEvent(engine->stream,
                reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(input_ready_event)), 0),
                "TensorRT input-ready event wait");
        if (engine->device_graph) {
            const auto status = cudaGraphLaunch(engine->device_graph, engine->stream);
            engine->device_failed = status != cudaSuccess;
            cuda_check(status, "TensorRT CUDA graph launch");
        } else if (!engine->context->enqueueV3(engine->stream)) {
            engine->device_failed = true;
            throw std::runtime_error("TensorRT device enqueueV3 failed; engine invalidated");
        }
        for (size_t i = 0; i < engine->outputs.size(); ++i) {
            const auto& slot = engine->outputs[i];
            auto& view = outputs[i];
            view = {};
            view.device_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(slot.device));
            view.nbytes = slot.spec.nbytes;
            view.rank = slot.spec.rank;
            view.dtype = slot.spec.dtype;
            for (uint32_t d = 0; d < view.rank; ++d) view.dimensions[d] = slot.spec.dimensions[d];
        }
        *output_count = static_cast<uint32_t>(engine->outputs.size());
        *stream_out = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(engine->stream));
        engine->device_pending = true;
        drain.armed = false;  // finish_device/destroy now own the drain obligation.
        return 0;
    } catch (const std::exception& error) {
        write_error(error_out, error_out_size, error.what());
        return 1;
    } catch (...) {
        write_error(error_out, error_out_size, "unexpected TensorRT device enqueue exception");
        return 2;
    }
}

extern "C" int novasight_tensorrt_finish_device(
    novasight_tensorrt_engine* engine, char* error_out, size_t error_out_size
) {
    try {
        if (engine == nullptr || !engine->device_pending || engine->device_failed)
            throw std::runtime_error("TensorRT has no valid outstanding device frame");
        const auto status = cudaStreamSynchronize(engine->stream);
        engine->device_pending = false;
        engine->device_failed = status != cudaSuccess;
        cuda_check(status, "TensorRT device frame completion");
        engine->warmed_up = true;
        return 0;
    } catch (const std::exception& error) {
        write_error(error_out, error_out_size, error.what());
        return 1;
    } catch (...) {
        write_error(error_out, error_out_size, "unexpected TensorRT device completion exception");
        return 2;
    }
}

extern "C" int novasight_tensorrt_capture_device_graph(
    novasight_tensorrt_engine* engine, char* error_out, size_t error_out_size
) {
    cudaGraph_t graph = nullptr;
    bool attempted = false;
    try {
        if (!engine || engine->device_pending || engine->device_failed || !engine->warmed_up)
            throw std::runtime_error("TensorRT graph requires a completed warm-up frame");
        if (engine->device_graph) return 0;
        attempted = true;
        cuda_check(cudaStreamBeginCapture(engine->stream, cudaStreamCaptureModeThreadLocal), "graph capture begin");
        const bool enqueued = engine->context->enqueueV3(engine->stream);
        // End capture even if TensorRT rejects it, so the stream exits capture mode.
        const auto ended = cudaStreamEndCapture(engine->stream, &graph);
        cuda_check(ended, "graph capture end");
        if (!enqueued || !graph) throw std::runtime_error("TensorRT graph capture failed");
        cuda_check(cudaGraphInstantiate(&engine->device_graph, graph, 0), "graph instantiate");
        cudaGraphDestroy(graph);
        return 0;
    } catch (const std::exception& error) {
        if (graph) cudaGraphDestroy(graph);
        if (attempted) engine->device_failed = true;
        write_error(error_out, error_out_size, error.what());
        return 1;
    } catch (...) {
        if (graph) cudaGraphDestroy(graph);
        if (attempted) engine->device_failed = true;
        write_error(error_out, error_out_size, "unexpected TensorRT graph capture exception");
        return 2;
    }
}

extern "C" int novasight_tensorrt_probe_zero(
    novasight_tensorrt_engine* engine,
    novasight_host_tensor_view* outputs,
    uint32_t output_capacity,
    uint32_t* output_count,
    char* error_out,
    size_t error_out_size
) {
    void* input_device = nullptr;
    try {
        if (engine == nullptr || engine->device_pending || engine->device_failed) {
            throw std::runtime_error("TensorRT zero probe engine is null, busy or failed");
        }
        const auto& input_spec = engine->spec.input;
        if (input_spec.nbytes > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("TensorRT zero probe input exceeds host size_t capacity");
        }
        cuda_check(
            cudaMalloc(&input_device, static_cast<size_t>(input_spec.nbytes)),
            "cudaMalloc zero probe input"
        );
        cuda_check(
            cudaMemsetAsync(
                input_device,
                0,
                static_cast<size_t>(input_spec.nbytes),
                engine->stream
            ),
            "cudaMemsetAsync zero probe input"
        );
        novasight_device_tensor_view input{};
        input.device_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(input_device));
        input.nbytes = input_spec.nbytes;
        input.rank = input_spec.rank;
        input.dtype = input_spec.dtype;
        for (uint32_t index = 0; index < input.rank; ++index) {
            input.dimensions[index] = static_cast<uint64_t>(input_spec.dimensions[index]);
        }
        const int result = execute_engine(
            engine,
            &input,
            outputs,
            output_capacity,
            output_count
        );
        cuda_check(cudaFree(input_device), "cudaFree zero probe input");
        return result;
    } catch (const std::exception& error) {
        if (input_device != nullptr) (void)cudaFree(input_device);
        if (output_count != nullptr) *output_count = 0;
        write_error(error_out, error_out_size, error.what());
        return 1;
    } catch (...) {
        if (input_device != nullptr) (void)cudaFree(input_device);
        if (output_count != nullptr) *output_count = 0;
        write_error(error_out, error_out_size, "unexpected TensorRT zero probe exception");
        return 2;
    }
}

extern "C" void novasight_tensorrt_destroy(novasight_tensorrt_engine* engine) {
    try {
        delete engine;
    } catch (...) {
    }
}
