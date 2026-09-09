#ifndef NOVASIGHT_GPU_FRAME_H
#define NOVASIGHT_GPU_FRAME_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define NOVASIGHT_GPU_FRAME_ABI 1U
typedef struct {
    uint32_t abi_version, width, height, candidates, classes;
    uint32_t channels_first, has_objectness, input_dtype, output_dtype, bgr, top_k, cuda_graph;
    float scale, confidence_threshold, nms_threshold;
} novasight_gpu_frame_config;
typedef struct { float left, top, width, height, confidence; uint32_t class_id; } novasight_gpu_detection;
typedef struct { uint32_t count, truncated; novasight_gpu_detection detections[256]; } novasight_gpu_result;
typedef struct novasight_gpu_frame novasight_gpu_frame;

/* One thread owns the context. Uses the existing TensorRT device API, never host outputs.
 * Input/output names and shape/dtype must match the admitted model manifest. */
int novasight_gpu_frame_create(const char* engine, const char* input_name, const char* output_name,
    const novasight_gpu_frame_config* config, novasight_gpu_frame** out, char* error, size_t error_size);
/* gst_buffer must remain alive for the call and contain one RGBA NVMM surface of model size.
 * Synchronous completion releases all GPU reads before return. Errors poison this context. */
int novasight_gpu_frame_process(novasight_gpu_frame* context, void* gst_buffer,
    novasight_gpu_result* out, char* error, size_t error_size);
void novasight_gpu_frame_destroy(novasight_gpu_frame* context);
#ifdef __cplusplus
}
#endif
#endif
