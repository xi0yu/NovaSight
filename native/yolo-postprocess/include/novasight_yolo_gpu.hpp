#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace novasight {

// Explicit raw YOLO contract, never inferred from a six-column tensor.
struct YoloGpuConfig {
    std::uint32_t candidates, classes, width, height;
    bool channels_first, has_objectness, float16;
    float confidence_threshold, nms_threshold;
    std::uint32_t top_k;  // Per class, AFTER NMS, as in the live nvinfer config.
};

struct YoloGpuDetection {
    float left, top, width, height, confidence;
    std::uint32_t class_id;
};

struct YoloGpuResult {
    // Same bounded prefix/truncation convention as the existing metadata bridge.
    static constexpr std::uint32_t capacity = 256;
    std::uint32_t count, truncated;
    YoloGpuDetection detections[capacity];
};

// One owner per model/session, one outstanding frame. No CPU postprocess path.
class YoloGpuPostprocessor {
public:
    explicit YoloGpuPostprocessor(YoloGpuConfig config);
    ~YoloGpuPostprocessor();
    YoloGpuPostprocessor(const YoloGpuPostprocessor&) = delete;
    YoloGpuPostprocessor& operator=(const YoloGpuPostprocessor&) = delete;

    // Producer work must precede this call on stream (or an explicit stream
    // event wait). Input and stream must remain alive through wait(). Scratch
    // and pinned final results are allocated once, not per frame.
    void enqueue(const void* device_tensor, std::size_t nbytes, cudaStream_t stream);
    // Throws on CUDA errors. Result is valid until the next enqueue/destruction;
    // the caller must copy final metadata into its owned, frame-stamped batch.
    const YoloGpuResult& wait();

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace novasight
