#ifndef NOVASIGHT_JETSON_PREPROCESS_NATIVE_H
#define NOVASIGHT_JETSON_PREPROCESS_NATIVE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NOVASIGHT_JETSON_PREPROCESS_ABI_VERSION 1

/*
 * Required ABI version hook. Python accepts a native shared library only when
 * this function returns NOVASIGHT_JETSON_PREPROCESS_ABI_VERSION, or when
 * novasight_status_json returns the same abi_version field.
 */
uint32_t novasight_abi_version(void);

/*
 * Prepare a model-ready GPU tensor from a JSON payload.
 *
 * payload_json contains JSON-safe metadata from the Python bridge. The Python
 * Gst.Sample resource_handle stays alive in Python while native preprocessing
 * runs. Native implementations must import either dmabuf_fd or gst_buffer_ptr;
 * gst_buffer_ptr is mapped only for the duration of the prepare call.
 *
 * On success, write a JSON object into result_json and return 0. The production
 * Python ctypes bridge requires device_ptr, nbytes, a positive release_token,
 * zero_copy=true, and a device memory_space value such as cuda_device. It may
 * contain shape, dtype, stream, backend, and reason.
 *
 * On failure, write a short JSON or plain text diagnostic into result_json and
 * return a non-zero value.
 */
int novasight_prepare_tensor_json(
    const char* payload_json,
    char* result_json,
    size_t result_json_size
);

/*
 * Required production readiness/status hook. Return 0 and write a JSON object.
 * A ready production library must explicitly report available/ready,
 * zero_copy=true, and a device memory_space. For example:
 * {"available":true,"ready":true,"backend":"jetson_cuda",
 *  "zero_copy":true,"memory_space":"cuda_device","abi_version":1}.
 *
 * Libraries that only export prepare without this readiness contract are
 * rejected before NVMM inference starts.
 */
int novasight_status_json(char* status_json, size_t status_json_size);

/*
 * Required production release hook for result.release_token returned by
 * prepare. Return 0 on success. Non-zero return codes are surfaced as runtime
 * release failures.
 */
int novasight_release_tensor(uint64_t release_token);

#ifdef __cplusplus
}
#endif

#endif
