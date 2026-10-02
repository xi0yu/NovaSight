#ifndef NOVASIGHT_TENSORRT_RUNTIME_H
#define NOVASIGHT_TENSORRT_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NOVASIGHT_TENSORRT_RUNTIME_ABI_VERSION 3U
#define NOVASIGHT_TENSORRT_MAX_TENSOR_NAME 128U
#define NOVASIGHT_TENSORRT_MAX_RANK 8U
#define NOVASIGHT_TENSORRT_MAX_OUTPUTS 8U
#define NOVASIGHT_TENSORRT_MAX_DEVICE_NAME 256U

typedef enum {
    NOVASIGHT_TENSOR_DTYPE_FLOAT32 = 1,
    NOVASIGHT_TENSOR_DTYPE_FLOAT16 = 2,
} novasight_tensor_dtype;

typedef struct {
    char name[NOVASIGHT_TENSORRT_MAX_TENSOR_NAME];
    uint32_t rank;
    int64_t dimensions[NOVASIGHT_TENSORRT_MAX_RANK];
    uint32_t dtype;
    uint64_t nbytes;
} novasight_tensor_spec;

typedef struct {
    novasight_tensor_spec input;
    uint32_t output_count;
    novasight_tensor_spec outputs[NOVASIGHT_TENSORRT_MAX_OUTPUTS];
    uint32_t input_dynamic;
    uint32_t selected_profile;
} novasight_engine_spec;

typedef struct {
    uint64_t device_ptr;
    uint64_t nbytes;
    uint32_t rank;
    uint64_t dimensions[NOVASIGHT_TENSORRT_MAX_RANK];
    uint32_t dtype;
} novasight_device_tensor_view;

typedef struct {
    const void* host_ptr;
    uint64_t nbytes;
    novasight_tensor_spec spec;
} novasight_host_tensor_view;

typedef struct {
    uint32_t runtime_abi_version;
    int32_t tensorrt_runtime_version;
    int32_t cuda_runtime_version;
    int32_t cuda_driver_version;
    int32_t device_ordinal;
    int32_t compute_capability_major;
    int32_t compute_capability_minor;
    uint32_t integrated;
    uint64_t total_global_memory;
    char device_name[NOVASIGHT_TENSORRT_MAX_DEVICE_NAME];
} novasight_tensorrt_environment;

typedef struct novasight_tensorrt_engine novasight_tensorrt_engine;

uint32_t novasight_tensorrt_abi_version(void);

int novasight_tensorrt_environment_query(
    novasight_tensorrt_environment* environment_out,
    char* error_out,
    size_t error_out_size
);

int novasight_tensorrt_create(
    const char* engine_path,
    const uint64_t* requested_input_shape,
    uint32_t requested_input_rank,
    novasight_tensorrt_engine** engine_out,
    novasight_engine_spec* spec_out,
    char* error_out,
    size_t error_out_size
);

int novasight_tensorrt_execute(
    novasight_tensorrt_engine* engine,
    const novasight_device_tensor_view* input,
    novasight_host_tensor_view* outputs,
    uint32_t output_capacity,
    uint32_t* output_count,
    char* error_out,
    size_t error_out_size
);

/* Additive device-output API; existing ABI-3 layouts and host API are unchanged.
 * Single-threaded owner: at most one outstanding enqueue per engine.
 * input_ready_event is a borrowed cudaEvent_t encoded as uint64_t; inference
 * waits for it on the GPU. Zero means the input is already ready. The event must
 * be recorded before enqueue and remain alive until finish_device returns.
 * Returns borrowed device views in spec.outputs order and a borrowed cudaStream_t
 * encoded as uint64_t. Queue GPU consumers on that stream; do not destroy it.
 * Input, views and engine must remain alive until finish_device returns. A second
 * enqueue or host execution is rejected until finish_device. No raw D2H copy.
 */
int novasight_tensorrt_enqueue_device(
    novasight_tensorrt_engine* engine,
    const novasight_device_tensor_view* input,
    uint64_t input_ready_event,
    novasight_device_tensor_view* outputs,
    uint32_t output_capacity,
    uint32_t* output_count,
    uint64_t* stream_out,
    char* error_out,
    size_t error_out_size
);

/* Drain inference and any consumers queued on the borrowed stream, then release
 * the outstanding frame. Completion failure poisons the engine; never publish its
 * results. destroy also drains outstanding device work before freeing storage.
 */
int novasight_tensorrt_finish_device(
    novasight_tensorrt_engine* engine,
    char* error_out,
    size_t error_out_size
);

/* Optional inference-only CUDA graph, captured after a completed warm-up frame.
 * The last input allocation must still be alive. Stable device enqueues replay
 * it; changing the input address or using the host API discards it BEFORE the
 * execution context is modified. Capture/launch failure invalidates the engine.
 * No model rewrite, CPU path, extra output copy or change to frame ownership.
 */
int novasight_tensorrt_capture_device_graph(
    novasight_tensorrt_engine* engine, char* error_out, size_t error_out_size
);

/* Execute one deterministic all-zero input owned by this runtime. This is
 * used only by the offline model-ingress probe and never by the live path. */
int novasight_tensorrt_probe_zero(
    novasight_tensorrt_engine* engine,
    novasight_host_tensor_view* outputs,
    uint32_t output_capacity,
    uint32_t* output_count,
    char* error_out,
    size_t error_out_size
);

void novasight_tensorrt_destroy(novasight_tensorrt_engine* engine);

#ifdef __cplusplus
}
#endif

#endif
