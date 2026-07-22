#ifndef NOVASIGHT_TENSORRT_RUNTIME_H
#define NOVASIGHT_TENSORRT_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NOVASIGHT_TENSORRT_RUNTIME_ABI_VERSION 1U
#define NOVASIGHT_TENSORRT_MAX_TENSOR_NAME 128U
#define NOVASIGHT_TENSORRT_MAX_RANK 8U
#define NOVASIGHT_TENSORRT_MAX_OUTPUTS 8U

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

typedef struct novasight_tensorrt_engine novasight_tensorrt_engine;

uint32_t novasight_tensorrt_abi_version(void);

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

void novasight_tensorrt_destroy(novasight_tensorrt_engine* engine);

#ifdef __cplusplus
}
#endif

#endif
