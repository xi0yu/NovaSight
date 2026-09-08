#include "novasight_yolo_gpu.hpp"
#include "novasight_deepstream_bridge.h"
#include "novasight_tensorrt_runtime.h"
#include <nvdsinfer_custom_impl.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

extern "C" bool NvDsInferParseNovaSightRaw(const std::vector<NvDsInferLayerInfo>&,
    const NvDsInferNetworkInfo&, const NvDsInferParseDetectionParams&,
    std::vector<NvDsInferObjectDetectionInfo>&);

using novasight::YoloGpuConfig;
using novasight::YoloGpuPostprocessor;
using novasight::YoloGpuResult;
using Box = NvDsInferObjectDetectionInfo;
static_assert(YoloGpuResult::capacity == NS_DS_MAX_DETECTIONS);

void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}
void cuda_ok(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class F> void rejects(F function) {
    try { function(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid operation was accepted");
}

struct Input {
    YoloGpuConfig cfg;
    std::vector<float> data;
    explicit Input(unsigned n = 1344, unsigned classes = 5, bool first = true,
                   bool objectness = false, bool half = false)
        : cfg{n, classes, 256, 256, first, objectness, half, 0.65f, 0.45f, 256},
          data(std::size_t(n) * (classes + (objectness ? 5 : 4)), 0) {}
    void set(unsigned row, unsigned channel, float value) {
        const auto channels = cfg.classes + (cfg.has_objectness ? 5 : 4);
        data[cfg.channels_first ? channel * cfg.candidates + row : row * channels + channel] = value;
    }
    void box(unsigned row, float cx, float cy, float w, float h, float score, unsigned label = 0) {
        set(row, 0, cx); set(row, 1, cy); set(row, 2, w); set(row, 3, h);
        if (cfg.has_objectness) set(row, 4, 1);
        set(row, (cfg.has_objectness ? 5 : 4) + label, score);
    }
};

// Independent sequential oracle: use the REAL NovaSight raw parser, followed
// by the observed DS 7.1 rules (stable score order per class, greedy IoU NMS,
// strict post-threshold > 0, then per-class Top-K). No GPU algorithm mirrored.
std::vector<Box> reference(const Input& input, const void* bytes) {
    const auto& c = input.cfg;
    NvDsInferLayerInfo layer{};
    layer.layerName = "output0";
    layer.buffer = const_cast<void*>(bytes);
    layer.dataType = c.float16 ? HALF : FLOAT;
    layer.inferDims.numDims = 3;
    layer.inferDims.d[0] = 1;
    const unsigned channels = c.classes + (c.has_objectness ? 5 : 4);
    layer.inferDims.d[1] = int(c.channels_first ? channels : c.candidates);
    layer.inferDims.d[2] = int(c.channels_first ? c.candidates : channels);
    NvDsInferParseDetectionParams params{};
    params.numClassesConfigured = c.classes;
    params.perClassPreclusterThreshold.assign(c.classes, c.confidence_threshold);
    std::vector<Box> candidates;
    require(NvDsInferParseNovaSightRaw({layer}, {c.width, c.height}, params, candidates), "CPU parser failed");
    std::vector<Box> output;
    for (unsigned label = 0; label < c.classes; ++label) {
        std::vector<Box> sorted;
        for (const auto& box : candidates) if (box.classId == label) sorted.push_back(box);
        std::stable_sort(sorted.begin(), sorted.end(), [](const Box& a, const Box& b) {
            return a.detectionConfidence > b.detectionConfidence;
        });
        std::vector<Box> survivors;
        for (const auto& box : sorted) {
            const bool suppressed = std::any_of(survivors.begin(), survivors.end(), [&](const Box& kept) {
                const float x = std::max(0.0f, std::min(box.left + box.width, kept.left + kept.width) - std::max(box.left, kept.left));
                const float y = std::max(0.0f, std::min(box.top + box.height, kept.top + kept.height) - std::max(box.top, kept.top));
                const float overlap = x * y;
                const float area = box.width * box.height + kept.width * kept.height - overlap;
                const float iou = area == 0 ? 0 : overlap / area;
                return !(iou <= c.nms_threshold);
            });
            if (!suppressed) survivors.push_back(box);
        }
        unsigned count = 0;
        for (const auto& box : survivors) {
            if (box.detectionConfidence > 0 && count < c.top_k) { output.push_back(box); ++count; }
        }
    }
    return output;
}

struct Trial {
    const Input& input;
    std::vector<__half> halves;
    void* device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    YoloGpuPostprocessor post;
    explicit Trial(const Input& data) : input(data), post(data.cfg) {
        if (data.cfg.float16) for (float f : data.data) halves.push_back(__float2half_rn(f));
        try {
            cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            cuda_ok(cudaEventCreate(&begin)); cuda_ok(cudaEventCreate(&end));
            cuda_ok(cudaMalloc(&device, size()));
            cuda_ok(cudaMemcpyAsync(device, bytes(), size(), cudaMemcpyHostToDevice, stream));
        } catch (...) { cleanup(); throw; }
    }
    ~Trial() { cleanup(); }
    void cleanup() {
        if (stream) cudaStreamSynchronize(stream);
        if (device) cudaFree(device);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (stream) cudaStreamDestroy(stream);
    }
    const void* bytes() const { return input.cfg.float16 ? static_cast<const void*>(halves.data()) : input.data.data(); }
    std::size_t size() const { return input.data.size() * (input.cfg.float16 ? 2 : 4); }
    YoloGpuResult run() {
        post.enqueue(device, size(), stream);
        return post.wait();
    }
};

void compare(const YoloGpuResult& actual, const std::vector<Box>& expected) {
    require(actual.count == std::min(expected.size(), std::size_t(YoloGpuResult::capacity)), "detection count mismatch");
    require(actual.truncated == expected.size() - actual.count, "truncation mismatch");
    for (unsigned i = 0; i < actual.count; ++i) {
        const auto& a = actual.detections[i]; const auto& b = expected[i];
        require(a.class_id == b.classId, "class/order mismatch");
        require(std::fabs(a.confidence - b.detectionConfidence) <= 1e-6f, "score mismatch");
        require(std::fabs(a.left - b.left) <= 1e-4f && std::fabs(a.top - b.top) <= 1e-4f
            && std::fabs(a.width - b.width) <= 1e-4f && std::fabs(a.height - b.height) <= 1e-4f, "geometry/tie-order mismatch");
    }
}

unsigned checks = 0;
YoloGpuResult verify(const Input& input) {
    Trial trial(input);
    const auto actual = trial.run();
    compare(actual, reference(input, trial.bytes()));
    ++checks;
    return actual;
}

void contracts() {
    for (bool first : {false, true}) for (bool obj : {false, true}) for (bool half : {false, true}) {
        Input input(67, 3, first, obj, half);
        input.cfg.confidence_threshold = 0.5f;
        input.cfg.nms_threshold = 0.3f;
        input.box(0, 5, 10, 10, 10, 0.9f);
        input.box(1, 10, 10, 10, 10, 0.8f);
        input.box(2, 15, 10, 10, 10, 0.7f); // A suppresses B; suppressed B must NOT suppress C.
        input.box(3, 5, 10, 10, 10, 0.75f, 1); // Class-aware, despite identical geometry.
        input.box(4, 100, 100, 10, 10, 0.5f, 2); // Equality passes confidence threshold.
        const auto result = verify(input);
        require(result.count == 4 && result.detections[1].left == 10, "greedy NMS chain failed");
        input.cfg.top_k = 1;
        require(verify(input).count == 3, "Top-K must be per class and after NMS");
    }
    Input ties(65);
    for (unsigned i = 0; i < 65; ++i) ties.box(i, 100 + i * 0.1f, 100, 30, 30, 0.8f);
    auto result = verify(ties);
    require(result.count == 1 && result.detections[0].left == 85, "equal-score stability failed");
    ties.cfg.nms_threshold = 1;
    require(verify(ties).count == 65, "IoU equality must not suppress");

    Input edge(64, 1);
    edge.cfg.nms_threshold = 0.5f;
    edge.box(0, 1.5f, 2, 3, 2, 0.9f); edge.box(1, 2.5f, 2, 3, 2, 0.8f);
    require(verify(edge).count == 2, "exact IoU threshold failed");
    edge.cfg.nms_threshold = std::nextafter(0.5f, 0.0f);
    require(verify(edge).count == 1, "IoU nextafter threshold failed");

    Input invalid(1344);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    invalid.box(0, 1, 1, 10, 10, 0.9f); // Clipping.
    invalid.box(1, 20, 20, -1, 5, 0.9f);
    invalid.box(2, nan, 20, 5, 5, 0.9f);
    invalid.box(3, 30, 30, 5, 5, inf);
    invalid.box(4, 40, 40, 5, 5, nan);
    invalid.box(5, 500, 500, 5, 5, 0.9f);
    invalid.box(6, 80, 80, 5, 5, 0.9f, 1);
    invalid.set(6, 4, nan); // NaN in another class does not erase a valid best class.
    require(verify(invalid).count == 2, "invalid value filtering failed");
    Input zeros(1, 1);
    zeros.cfg.confidence_threshold = 0;
    zeros.box(0, 5, 5, 4, 4, 0);
    require(verify(zeros).count == 0, "SDK strict post-cluster threshold failed");
    verify(Input());

    std::mt19937 rng(20260908);
    for (unsigned pass = 0; pass < 16; ++pass) {
        Input input(pass & 1 ? 1344 : 8400, 5, pass & 2, pass & 4, pass & 8);
        for (unsigned i = 0; i < input.cfg.candidates; ++i) {
            input.box(i, float(int(rng() % 300) - 20), float(int(rng() % 300) - 20),
                float(1 + rng() % 80), float(1 + rng() % 80), float(rng() % 1025) / 1024, rng() % 5);
            if (input.cfg.has_objectness) input.set(i, 4, float(rng() % 1025) / 1024);
        }
        verify(input);
    }
    Input many(1344);
    for (unsigned i = 0; i < 1000; ++i)
        many.box(i, float(2 + i % 50 * 5), float(2 + i / 50 * 5), 1, 1, 0.9f, i % 5);
    result = verify(many);
    require(result.count == 256 && result.truncated == 744, "bounded prefix/truncation failed");

    Trial trial(many);
    rejects([&] { trial.post.wait(); });
    rejects([&] { trial.post.enqueue(nullptr, trial.size(), trial.stream); });
    rejects([&] { trial.post.enqueue(trial.device, trial.size() - 1, trial.stream); });
    rejects([&] { trial.post.enqueue(trial.bytes(), trial.size(), trial.stream); });
    trial.post.enqueue(trial.device, trial.size(), trial.stream);
    rejects([&] { trial.post.enqueue(trial.device, trial.size(), trial.stream); });
    compare(trial.post.wait(), reference(many, trial.bytes()));
    cuda_ok(cudaMemsetAsync(trial.device, 0, trial.size(), trial.stream));
    require(trial.run().count == 0, "stale results survived workspace reuse");
    auto bad = many.cfg;
    bad.candidates = 32769; rejects([&] { YoloGpuPostprocessor post(bad); });
    bad = many.cfg; bad.nms_threshold = nan; rejects([&] { YoloGpuPostprocessor post(bad); });
    bad = many.cfg; bad.top_k = 0; rejects([&] { YoloGpuPostprocessor post(bad); });
}

double percentile(std::vector<double> data, double fraction) {
    std::sort(data.begin(), data.end());
    return data[std::size_t(std::ceil(data.size() * fraction)) - 1];
}
void benchmark(const char* name, Input input) {
    Trial trial(input);
    const auto expected = reference(input, trial.bytes());
    for (int i = 0; i < 20; ++i) compare(trial.run(), expected);
    std::vector<double> cpu, wall, gpu;
    for (int i = 0; i < 100; ++i) {
        auto start = std::chrono::steady_clock::now();
        const auto actual = trial.run();
        wall.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
        compare(actual, expected);
        start = std::chrono::steady_clock::now();
        const auto output = reference(input, trial.bytes());
        cpu.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
        require(output.size() == expected.size(), "CPU benchmark changed results");
        cuda_ok(cudaEventRecord(trial.begin, trial.stream));
        trial.post.enqueue(trial.device, trial.size(), trial.stream);
        cuda_ok(cudaEventRecord(trial.end, trial.stream));
        trial.post.wait();
        cuda_ok(cudaEventSynchronize(trial.end));
        float ms = 0; cuda_ok(cudaEventElapsedTime(&ms, trial.begin, trial.end));
        gpu.push_back(ms * 1000);
    }
    std::cout << "{\"case\":\"" << name << "\",\"synthetic\":true,\"samples\":100,\"raw_candidates\":"
        << input.cfg.candidates << ",\"survivors\":" << expected.size()
        << ",\"cpu_reference_p50_us\":" << percentile(cpu, .5) << ",\"cpu_reference_p95_us\":" << percentile(cpu, .95)
        << ",\"gpu_host_wall_p50_us\":" << percentile(wall, .5) << ",\"gpu_host_wall_p95_us\":" << percentile(wall, .95)
        << ",\"gpu_stream_p95_us\":" << percentile(gpu, .95) << ",\"d2h_bytes\":" << sizeof(YoloGpuResult) << "}\n";
}

void engine_check(const char* path, bool trace_only) {
    novasight_tensorrt_engine* raw = nullptr;
    novasight_engine_spec spec{};
    char error[1024]{};
    auto status_ok = [&](int status) { if (status) throw std::runtime_error(error); };
    status_ok(novasight_tensorrt_create(path, nullptr, 0, &raw, &spec, error, sizeof(error)));
    // Engine owner must drain before its externally owned input can be freed.
    std::unique_ptr<novasight_tensorrt_engine, decltype(&novasight_tensorrt_destroy)>
        engine(raw, novasight_tensorrt_destroy);
    require(spec.output_count == 1 && spec.input.dimensions[2] == 256 && spec.input.dimensions[3] == 256,
        "This engine check requires the measured 256px raw YOLO fixture");
    const auto& output = spec.outputs[0];
    require(output.rank == 3 && output.dimensions[0] == 1 && output.dimensions[1] == 9
        && output.dimensions[2] == 1344 && output.dtype == NOVASIGHT_TENSOR_DTYPE_FLOAT32,
        "This engine check requires explicit [1,9,1344] FP32 output, five classes, no objectness");
    void* input_memory = nullptr;
    cuda_ok(cudaMalloc(&input_memory, spec.input.nbytes));
    cudaStream_t producer = nullptr;
    cudaEvent_t input_ready = nullptr;
    const auto cleanup = [&] {
        engine.reset();
        if (producer) cudaStreamSynchronize(producer);
        if (input_ready) cudaEventDestroy(input_ready);
        if (producer) cudaStreamDestroy(producer);
        cudaFree(input_memory);
    };
    try {
        cuda_ok(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
        cuda_ok(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming));
        cuda_ok(cudaMemsetAsync(input_memory, 0, spec.input.nbytes, producer));
        cuda_ok(cudaEventRecord(input_ready, producer));
        const auto ready = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(input_ready));
        novasight_device_tensor_view input{};
        input.device_ptr = reinterpret_cast<std::uintptr_t>(input_memory);
        input.nbytes = spec.input.nbytes; input.rank = spec.input.rank; input.dtype = spec.input.dtype;
        for (unsigned d = 0; d < input.rank; ++d) input.dimensions[d] = spec.input.dimensions[d];
        novasight_device_tensor_view device_outputs[NOVASIGHT_TENSORRT_MAX_OUTPUTS]{};
        novasight_host_tensor_view host_outputs[NOVASIGHT_TENSORRT_MAX_OUTPUTS]{};
        unsigned count = 0;
        std::uint64_t stream = 0;
        YoloGpuPostprocessor post(Input().cfg);
        std::vector<Box> expected;
        if (!trace_only) {
            cuda_ok(cudaEventSynchronize(input_ready)); // Old host API has no event argument.
            status_ok(novasight_tensorrt_execute(engine.get(), &input, host_outputs,
                NOVASIGHT_TENSORRT_MAX_OUTPUTS, &count, error, sizeof(error)));
            expected = reference(Input(), host_outputs[0].host_ptr);
            require(novasight_tensorrt_finish_device(engine.get(), error, sizeof(error)) != 0, "finish without enqueue accepted");
            count = 99; stream = 99;
            require(novasight_tensorrt_enqueue_device(engine.get(), &input, ready, device_outputs,
                0, &count, &stream, error, sizeof(error)) != 0 && count == 0 && stream == 0,
                "device output capacity guard failed");
        }
        std::vector<double> times;
        unsigned detections = 0;
        for (unsigned iteration = 0; iteration < 120; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            status_ok(novasight_tensorrt_enqueue_device(engine.get(), &input, ready, device_outputs,
                NOVASIGHT_TENSORRT_MAX_OUTPUTS, &count, &stream, error, sizeof(error)));
            require(count == 1 && device_outputs[0].nbytes == output.nbytes, "device output spec mismatch");
            if (!trace_only && iteration == 0) {
                unsigned rejected_count = 99; std::uint64_t rejected_stream = 99;
                require(novasight_tensorrt_enqueue_device(engine.get(), &input, ready, device_outputs,
                    NOVASIGHT_TENSORRT_MAX_OUTPUTS, &rejected_count, &rejected_stream, error, sizeof(error)) != 0
                    && rejected_count == 0 && rejected_stream == 0, "outstanding device frame overwritten");
                require(novasight_tensorrt_execute(engine.get(), &input, host_outputs,
                    NOVASIGHT_TENSORRT_MAX_OUTPUTS, &rejected_count, error, sizeof(error)) != 0,
                    "host execution overwrote an outstanding device frame");
            }
            post.enqueue(reinterpret_cast<void*>(device_outputs[0].device_ptr), device_outputs[0].nbytes,
                reinterpret_cast<cudaStream_t>(stream));
            const auto& result = post.wait();
            status_ok(novasight_tensorrt_finish_device(engine.get(), error, sizeof(error)));
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
            if (!trace_only) compare(result, expected);
            detections = result.count;
            if (iteration >= 20) times.push_back(us);
        }
        if (!trace_only) {
            // Old offline ABI remains usable after the device frame is released.
            status_ok(novasight_tensorrt_execute(engine.get(), &input, host_outputs,
                NOVASIGHT_TENSORRT_MAX_OUTPUTS, &count, error, sizeof(error)));
            require(reference(Input(), host_outputs[0].host_ptr).size() == expected.size(), "host API changed after device use");
            status_ok(novasight_tensorrt_enqueue_device(engine.get(), &input, ready, device_outputs,
                NOVASIGHT_TENSORRT_MAX_OUTPUTS, &count, &stream, error, sizeof(error)));
            engine.reset(); // Exercise drain-on-destroy with outstanding inference.
        }
        std::cout << "{\"engine_gpu_pass\":true,\"zero_input\":true,\"trace_only\":" << (trace_only ? "true" : "false")
            << ",\"samples\":100,\"detections\":" << detections << ",\"infer_post_host_p50_us\":" << percentile(times, .5)
            << ",\"infer_post_host_p95_us\":" << percentile(times, .95) << ",\"final_d2h_bytes\":" << sizeof(YoloGpuResult) << "}\n";
        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }
}

int main(int argc, char** argv) {
    try {
        if (argc == 3 && (std::strcmp(argv[1], "--engine") == 0 || std::strcmp(argv[1], "--trace-engine") == 0)) {
            engine_check(argv[2], std::strcmp(argv[1], "--trace-engine") == 0);
        } else if (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0) {
            benchmark("empty_1344", Input());
            Input sparse;
            for (unsigned i = 0; i < 20; ++i) sparse.box(i, 10 + i * 8, 100, 10, 10, .8f, i % 5);
            benchmark("sparse_1344", sparse);
            Input dense;
            std::mt19937 rng(20260908);
            for (unsigned i = 0; i < 1344; ++i)
                dense.box(i, float(rng() % 256), float(rng() % 256), 20, 20, .65f + float(rng() % 350) / 1000, i % 5);
            benchmark("dense_1344", dense);
        } else {
            contracts();
            std::cout << "YOLO_GPU_CONTRACT_PASS cases=" << checks << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "YOLO_GPU_CONTRACT_FAIL: " << error.what() << '\n';
        return 1;
    }
}
