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

/* Host-side wall-clock stage timings of one novasight_gpu_frame_process call.
 * All durations are nanoseconds measured with std::chrono::steady_clock at the
 * synchronous boundaries of the asynchronous CUDA pipeline, so a stall shows
 * up in exactly one stage:
 *   import_ns       gst_buffer_map + NvBufSurface checks + EGLImage register/map
 *   launch_ns       preprocess/TRT/decode-NMS kernel enqueue (host overhead)
 *   gpu_wait_ns     post->wait + finish: blocks until preprocess+TensorRT+NMS done
 *   release_ns      producer stream sync + EGL unregister/unmap + gst_buffer_unmap
 *   graph_capture_ns  CUDA graph capture on the first warmed frame (0 afterwards)
 *   total_ns        whole call, from entry to just before the result memcpy
 * warmed=1 once the CUDA graph has been captured. Zeroed on error. */
#define NOVASIGHT_GPU_TIMINGS_ABI 1U
typedef struct {
    uint32_t abi_version, warmed;
    uint64_t total_ns, import_ns, launch_ns, gpu_wait_ns, release_ns, graph_capture_ns;
} novasight_gpu_timings;

typedef struct novasight_gpu_frame novasight_gpu_frame;

/* One thread owns the context. Uses the existing TensorRT device API, never host outputs.
 * Input/output names and shape/dtype must match the admitted model manifest. */
int novasight_gpu_frame_create(const char* engine, const char* input_name, const char* output_name,
    const novasight_gpu_frame_config* config, novasight_gpu_frame** out, char* error, size_t error_size);
/* gst_buffer must remain alive for the call and contain one RGBA NVMM surface of model size.
 * Synchronous completion releases all GPU reads before return. Errors poison this context.
 * timings may be NULL; otherwise it is zeroed and filled only on success. */
int novasight_gpu_frame_process(novasight_gpu_frame* context, void* gst_buffer,
    novasight_gpu_result* out, novasight_gpu_timings* timings, char* error, size_t error_size);
void novasight_gpu_frame_destroy(novasight_gpu_frame* context);
#ifdef __cplusplus
}
#endif
#endif
