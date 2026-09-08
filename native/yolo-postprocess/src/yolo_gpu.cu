#include "novasight_yolo_gpu.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace novasight {
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template<class T> void allocate(T*& pointer, std::size_t count) {
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), "YOLO cudaMalloc");
}

template<class T> __device__ float value(const T* input, unsigned i) {
    return static_cast<float>(input[i]);
}
template<> __device__ float value(const __half* input, unsigned i) {
    return __half2float(input[i]);
}

template<class T>
__global__ void decode(const T* input, YoloGpuConfig cfg,
                       YoloGpuDetection* boxes, float* scores, unsigned* indices) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.candidates) return;
    scores[i] = -CUDART_INF_F;
    indices[i] = i;
    boxes[i] = {};
    const unsigned channels = cfg.classes + (cfg.has_objectness ? 5 : 4);
    const unsigned stride = cfg.channels_first ? cfg.candidates : 1;
    const T* row = input + (cfg.channels_first ? i : i * channels);
    const float cx = value(row, 0), cy = value(row, stride);
    const float width = value(row, 2 * stride), height = value(row, 3 * stride);
    if (!isfinite(cx) || !isfinite(cy) || !isfinite(width) || !isfinite(height)
        || width <= 0 || height <= 0) return;
    const float objectness = cfg.has_objectness ? value(row, 4 * stride) : 1.0f;
    float best = -CUDART_INF_F;
    unsigned label = 0;
    for (unsigned c = 0; c < cfg.classes; ++c) {
        const float score = objectness * value(row, (channels - cfg.classes + c) * stride);
        if (score > best) { best = score; label = c; }
    }
    if (!isfinite(best) || best < cfg.confidence_threshold) return;
    const float left = fmaxf(0, cx - width * 0.5f);
    const float top = fmaxf(0, cy - height * 0.5f);
    const float right = fminf(float(cfg.width), cx + width * 0.5f);
    const float bottom = fminf(float(cfg.height), cy + height * 0.5f);
    if (right <= left || bottom <= top) return;
    boxes[i] = {left, top, right - left, bottom - top, best, label};
    scores[i] = best;
}

__device__ float iou(const YoloGpuDetection& a, const YoloGpuDetection& b) {
    const float x = fmaxf(0, fminf(a.left + a.width, b.left + b.width) - fmaxf(a.left, b.left));
    const float y = fmaxf(0, fminf(a.top + a.height, b.top + b.height) - fmaxf(a.top, b.top));
    const float overlap = x * y;
    const float area = a.width * a.height + b.width * b.height - overlap;
    return area == 0 ? 0 : overlap / area;
}

__global__ void select_survivors(const YoloGpuDetection* boxes, const float* scores,
                                const unsigned* order, float threshold, unsigned n,
                                unsigned top_k, unsigned* counts, unsigned* kept) {
    __shared__ YoloGpuDetection survivors[YoloGpuResult::capacity];
    __shared__ YoloGpuDetection candidates[256];
    __shared__ unsigned originals[256], matches[8];
    unsigned count = 0;
    for (unsigned base = 0; base < n && count < top_k; base += 256) {
        const unsigned position = base + threadIdx.x;
        const bool valid = position < n && scores[position] > 0; // SDK post-threshold > 0.
        const unsigned original = valid ? order[position] : 0;
        const auto row = boxes[original];
        candidates[threadIdx.x] = row;
        originals[threadIdx.x] = original;
        const unsigned ballot = __ballot_sync(0xffffffff, valid && row.class_id == blockIdx.x);
        if (threadIdx.x % 32 == 0) matches[threadIdx.x / 32] = ballot;
        if (!__syncthreads_or(valid)) break;
        // Candidates remain in stable score order. Compare only to earlier
        // survivors, across the block, rather than materializing all pairwise IoUs.
        for (unsigned group = 0; group < 8 && count < top_k; ++group) {
            unsigned remaining = matches[group];
            while (remaining && count < top_k) {
                const unsigned index = group * 32 + __ffs(remaining) - 1;
                remaining &= remaining - 1;
                const auto candidate = candidates[index];
                const bool suppressed = threadIdx.x < count
                    && !(iou(candidate, survivors[threadIdx.x]) <= threshold);
                if (__syncthreads_or(suppressed)) continue;
                if (threadIdx.x == 0) {
                    kept[blockIdx.x * top_k + count] = originals[index];
                    survivors[count] = candidate;
                }
                ++count;
                __syncthreads();
            }
        }
        __syncthreads(); // All readers finish before the next candidate tile overwrites it.
    }
    // Exact prefix of full greedy NMS followed by per-class Top-K: later
    // lower-scoring candidates cannot suppress an earlier survivor.
    if (threadIdx.x == 0) counts[blockIdx.x] = count;
}

__global__ void pack_result(const YoloGpuDetection* boxes, const unsigned* kept,
                           const unsigned* counts, unsigned classes, unsigned top_k,
                           YoloGpuResult* result) {
    const unsigned i = threadIdx.x;
    unsigned offset = 0;
    YoloGpuDetection detection{};
    for (unsigned c = 0; c < classes; ++c) {
        if (i >= offset && i - offset < counts[c])
            detection = boxes[kept[c * top_k + i - offset]];
        offset += counts[c];
    }
    result->detections[i] = detection;
    if (i == 0) {
        result->count = min(offset, YoloGpuResult::capacity);
        result->truncated = offset - result->count;
    }
}
}  // namespace

struct YoloGpuPostprocessor::State {
    YoloGpuConfig config;
    YoloGpuDetection* boxes = nullptr;
    float *scores = nullptr, *sorted_scores = nullptr;
    unsigned *indices = nullptr, *order = nullptr, *counts = nullptr, *kept = nullptr;
    void* sort_storage = nullptr;
    std::size_t sort_bytes = 0, input_bytes = 0;
    YoloGpuResult *device_result = nullptr, *host_result = nullptr;
    cudaEvent_t done = nullptr;
    cudaStream_t stream = nullptr;
    int device = 0;
    bool pending = false, failed = false;

    explicit State(YoloGpuConfig c) : config(c) {}
    ~State() {
        if (pending) (void)cudaStreamSynchronize(stream);
        if (done) cudaEventDestroy(done);
        cudaFree(boxes); cudaFree(scores); cudaFree(sorted_scores);
        cudaFree(indices); cudaFree(order); cudaFree(counts); cudaFree(kept);
        cudaFree(sort_storage); cudaFree(device_result);
        if (host_result) cudaFreeHost(host_result);
    }
};

YoloGpuPostprocessor::YoloGpuPostprocessor(YoloGpuConfig config)
    : state_(std::make_unique<State>(config)) {
    // Bounded raw contract for this candidate; larger models need separate validation.
    if (config.candidates == 0 || config.candidates > 32768 || config.classes == 0
        || config.classes > 1024 || config.width == 0 || config.width > 16384
        || config.height == 0 || config.height > 16384 || config.top_k == 0
        || config.top_k > YoloGpuResult::capacity
        || !std::isfinite(config.confidence_threshold) || config.confidence_threshold < 0
        || config.confidence_threshold > 1 || !std::isfinite(config.nms_threshold)
        || config.nms_threshold < 0 || config.nms_threshold > 1)
        throw std::invalid_argument("Unsupported raw YOLO GPU contract");
    auto& s = *state_;
    check(cudaGetDevice(&s.device), "YOLO owning device");
    const unsigned n = config.candidates;
    s.input_bytes = std::size_t(n) * (config.classes + (config.has_objectness ? 5 : 4))
        * (config.float16 ? 2 : 4);
    allocate(s.boxes, n); allocate(s.scores, n); allocate(s.sorted_scores, n);
    allocate(s.indices, n); allocate(s.order, n); allocate(s.counts, config.classes);
    allocate(s.kept, std::size_t(config.classes) * config.top_k);
    allocate(s.device_result, 1);
    check(cudaMallocHost(reinterpret_cast<void**>(&s.host_result), sizeof(YoloGpuResult)), "YOLO pinned result");
    check(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming), "YOLO event create");
    check(cub::DeviceRadixSort::SortPairsDescending(nullptr, s.sort_bytes,
        s.scores, s.sorted_scores, s.indices, s.order, n), "YOLO sort sizing");
    check(cudaMalloc(&s.sort_storage, s.sort_bytes), "YOLO sort scratch");
}

YoloGpuPostprocessor::~YoloGpuPostprocessor() = default;

void YoloGpuPostprocessor::enqueue(const void* input, std::size_t nbytes, cudaStream_t stream) {
    auto& s = *state_;
    if (s.failed || s.pending) throw std::logic_error("YOLO session failed or has an outstanding frame");
    if (!input || nbytes != s.input_bytes
        || reinterpret_cast<std::uintptr_t>(input) % (s.config.float16 ? 2 : 4))
        throw std::invalid_argument("YOLO device tensor does not match the configured contract");
    cudaPointerAttributes attributes{};
    check(cudaPointerGetAttributes(&attributes, input), "YOLO input memory type");
    int current_device = 0;
    check(cudaGetDevice(&current_device), "YOLO current device");
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != s.device
        || current_device != s.device)
        throw std::invalid_argument("YOLO input and caller must use the owning CUDA device");
    s.stream = stream;
    s.pending = true;
    try {
        const unsigned n = s.config.candidates;
        if (s.config.float16)
            decode<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const __half*>(input), s.config, s.boxes, s.scores, s.indices);
        else
            decode<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const float*>(input), s.config, s.boxes, s.scores, s.indices);
        check(cudaGetLastError(), "YOLO decode launch");
        check(cub::DeviceRadixSort::SortPairsDescending(s.sort_storage, s.sort_bytes,
            s.scores, s.sorted_scores, s.indices, s.order, n, 0, 32, stream), "YOLO stable sort");
        select_survivors<<<s.config.classes, 256, 0, stream>>>(s.boxes,
            s.sorted_scores, s.order, s.config.nms_threshold, n, s.config.top_k, s.counts, s.kept);
        check(cudaGetLastError(), "YOLO NMS selection launch");
        pack_result<<<1, YoloGpuResult::capacity, 0, stream>>>(s.boxes, s.kept,
            s.counts, s.config.classes, s.config.top_k, s.device_result);
        check(cudaGetLastError(), "YOLO packing launch");
        check(cudaMemcpyAsync(s.host_result, s.device_result, sizeof(YoloGpuResult),
            cudaMemcpyDeviceToHost, stream), "YOLO final results D2H");
        check(cudaEventRecord(s.done, stream), "YOLO completion record");
    } catch (...) {
        s.failed = true;
        // Drain before freeing/reusing buffers, including partially submitted work.
        (void)cudaStreamSynchronize(stream);
        s.pending = false;
        throw;
    }
}

const YoloGpuResult& YoloGpuPostprocessor::wait() {
    auto& s = *state_;
    if (s.failed || !s.pending) throw std::logic_error("YOLO has no readable pending result");
    const auto status = cudaEventSynchronize(s.done);
    if (status != cudaSuccess) { s.failed = true; check(status, "YOLO completion wait"); }
    s.pending = false;
    return *s.host_result;
}
}  // namespace novasight
