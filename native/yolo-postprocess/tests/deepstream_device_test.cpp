#include "novasight_yolo_gpu.hpp"
#include "novasight_tensorrt_runtime.h"
#include <nvdsinfer_context.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

void require(bool condition, const char* error) {
    if (!condition) throw std::runtime_error(error);
}
void cuda_ok(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void ds_ok(NvDsInferStatus status) {
    if (status != NVDSINFER_SUCCESS) throw std::runtime_error(NvDsInferStatus2Str(status));
}

struct Fixture {
    NvDsInferContextHandle context = nullptr;
    void* input = nullptr;
    cudaStream_t consumer = nullptr;
    NvDsInferContextBatchOutput batch{};
    bool borrowed = false;
    ~Fixture() {
        if (consumer) cudaStreamSynchronize(consumer);
        if (context) {
            if (borrowed) context->releaseBatchOutput(batch);
            context->destroy(); // The SDK must release input before we free it.
        }
        if (input) cudaFree(input);
        if (consumer) cudaStreamDestroy(consumer);
    }
};

std::vector<float> direct_reference(const char* path, void* input) {
    novasight_tensorrt_engine* raw = nullptr;
    novasight_engine_spec spec{};
    char error[1024]{};
    require(novasight_tensorrt_create(path, nullptr, 0, &raw, &spec, error, sizeof(error)) == 0, error);
    std::unique_ptr<novasight_tensorrt_engine, decltype(&novasight_tensorrt_destroy)>
        engine(raw, novasight_tensorrt_destroy);
    require(spec.input.dtype == NOVASIGHT_TENSOR_DTYPE_FLOAT32 && spec.input.nbytes == 3 * 256 * 256 * sizeof(float)
        && spec.output_count == 1 && spec.outputs[0].dtype == NOVASIGHT_TENSOR_DTYPE_FLOAT32
        && spec.outputs[0].rank == 3 && spec.outputs[0].dimensions[0] == 1
        && spec.outputs[0].dimensions[1] == 9 && spec.outputs[0].dimensions[2] == 1344,
        "Reference engine differs from the explicit measured fixture contract");
    novasight_device_tensor_view view{};
    view.device_ptr = reinterpret_cast<std::uintptr_t>(input);
    view.nbytes = spec.input.nbytes; view.rank = spec.input.rank; view.dtype = spec.input.dtype;
    for (unsigned d = 0; d < view.rank; ++d) view.dimensions[d] = spec.input.dimensions[d];
    novasight_host_tensor_view output[NOVASIGHT_TENSORRT_MAX_OUTPUTS]{};
    unsigned count = 0;
    require(novasight_tensorrt_execute(engine.get(), &view, output, NOVASIGHT_TENSORRT_MAX_OUTPUTS,
        &count, error, sizeof(error)) == 0, error);
    const auto* values = static_cast<const float*>(output[0].host_ptr);
    return {values, values + 9 * 1344};
}

int main(int argc, char** argv) {
    try {
        require(argc == 2 || (argc == 3 && std::strcmp(argv[1], "--trace") == 0),
            "Usage: deepstream_device_test [--trace] ENGINE");
        const bool trace = argc == 3;
        const char* path = argv[argc - 1];
        Fixture fixture;
        cuda_ok(cudaMalloc(&fixture.input, 3 * 256 * 256 * sizeof(float)));
        cuda_ok(cudaMemset(fixture.input, 0, 3 * 256 * 256 * sizeof(float)));
        cuda_ok(cudaStreamCreateWithFlags(&fixture.consumer, cudaStreamNonBlocking));
        const auto expected = trace ? std::vector<float>{} : direct_reference(path, fixture.input);
        NvDsInferContextInitParams init{};
        NvDsInferContext_ResetInitParams(&init);
        require(std::strlen(path) < sizeof(init.modelEngineFilePath), "Engine path exceeds SDK capacity");
        std::strcpy(init.modelEngineFilePath, path);
        init.uniqueID = 1;
        init.gpuID = 0;
        init.maxBatchSize = 1;
        init.networkType = NvDsInferNetworkType_Other;
        init.inputFromPreprocessedTensor = 1;
        init.disableOutputHostCopy = 1;
        init.copyInputToHostBuffers = 0;
        init.outputBufferPoolSize = 2;
        init.autoIncMem = 0;
        ds_ok(createNvDsInferContext(&fixture.context, init, nullptr,
            [](NvDsInferContextHandle, unsigned, NvDsInferLogLevel, const char* message, void*) {
                if (message) std::fprintf(stderr, "%s\n", message);
            }));
        std::vector<NvDsInferLayerInfo> layers;
        fixture.context->fillLayersInfo(layers);
        require(layers.size() == 2, "Expected one input and one output binding");
        const auto input_it = std::find_if(layers.begin(), layers.end(), [](const auto& layer) { return layer.isInput; });
        const auto output_it = std::find_if(layers.begin(), layers.end(), [](const auto& layer) { return !layer.isInput; });
        require(input_it != layers.end() && output_it != layers.end(), "Missing input/output binding");
        require(input_it->dataType == FLOAT && input_it->inferDims.numDims == 3
            && input_it->inferDims.d[0] == 3 && input_it->inferDims.d[1] == 256 && input_it->inferDims.d[2] == 256
            && output_it->dataType == FLOAT && output_it->inferDims.numDims == 2
            && output_it->inferDims.d[0] == 9 && output_it->inferDims.d[1] == 1344,
            "DeepStream engine differs from the explicit measured fixture contract");
        NvDsInferLayerInfo input = *input_it;
        input.buffer = fixture.input;
        // The preprocessed-tensor API includes batch, whereas fillLayersInfo does not.
        input.inferDims.numDims = 4;
        input.inferDims.d[0] = 1; input.inferDims.d[1] = 3;
        input.inferDims.d[2] = 256; input.inferDims.d[3] = 256;
        input.inferDims.numElements = 3 * 256 * 256;
        NvDsInferContextBatchPreprocessedInput batch_input{};
        batch_input.tensors = &input;
        batch_input.numInputTensors = 1;
        // Input is immutable throughout the test and lives beyond context destruction.
        novasight::YoloGpuPostprocessor post({1344, 5, 256, 256, true, false, false, .65f, .45f, 256});
        std::vector<double> times;
        unsigned detections = 0;
        for (unsigned iteration = 0; iteration < 120; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            ds_ok(fixture.context->queueInputBatchPreprocessed(batch_input));
            ds_ok(fixture.context->dequeueOutputBatch(fixture.batch));
            fixture.borrowed = true;
            require(fixture.batch.numOutputDeviceBuffers == 1 && fixture.batch.numFrames == 1,
                "Unexpected DeepStream device batch");
            void* device = fixture.batch.outputDeviceBuffers[0];
            // Dequeue synchronizes the SDK's output-ready event. Keep its batch
            // borrowed until our GPU consumer and final result copy finish.
            post.enqueue(device, 9 * 1344 * sizeof(float), fixture.consumer);
            const auto& result = post.wait();
            detections = result.count;
            if (!trace && iteration == 0) {
                std::vector<float> actual(expected.size());
                cuda_ok(cudaMemcpy(actual.data(), device, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (std::size_t i = 0; i < actual.size(); ++i)
                    require(std::isfinite(actual[i]) && std::fabs(actual[i] - expected[i])
                        <= 1e-5f * std::max(1.0f, std::fabs(expected[i])), "DeepStream/direct TensorRT raw output mismatch");
            }
            fixture.context->releaseBatchOutput(fixture.batch);
            fixture.borrowed = false;
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
            if (iteration >= 20) times.push_back(us);
        }
        std::sort(times.begin(), times.end());
        std::cout << "{\"deepstream_device_pass\":true,\"zero_input\":true,\"trace_only\":" << (trace ? "true" : "false")
            << ",\"samples\":100,\"detections\":" << detections << ",\"infer_post_host_p50_us\":" << times[49]
            << ",\"infer_post_host_p95_us\":" << times[94] << ",\"final_d2h_bytes\":" << sizeof(novasight::YoloGpuResult) << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DEEPSTREAM_DEVICE_FAIL: " << error.what() << '\n';
        return 1;
    }
}
