#include "novasight_jetson_preprocess_native.h"
#include "novasight_jetson_preprocess_native_jetson_support.h"

#include <EGL/egl.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvbufsurface.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

using novasight::jetson_preprocess::TensorRequest;

struct DeviceAllocation {
    void* ptr = nullptr;
    uint64_t nbytes = 0;
};

std::mutex g_allocations_mutex;
std::unordered_map<uint64_t, DeviceAllocation> g_allocations;
uint64_t g_next_release_token = 1;

__device__ unsigned char clamp_u8(int value) {
    if (value < 0) {
        return 0;
    }
    if (value > 255) {
        return 255;
    }
    return static_cast<unsigned char>(value);
}

__device__ int scaled_source_index(int dst_index, int dst_size, int src_size) {
    int value = (dst_index * src_size) / dst_size;
    if (value < 0) {
        return 0;
    }
    if (value >= src_size) {
        return src_size - 1;
    }
    return value;
}

__device__ void write_tensor_value(
    void* output,
    int dtype_code,
    size_t index,
    float value
) {
    if (dtype_code == 16) {
        reinterpret_cast<__half*>(output)[index] = __float2half(value);
    } else {
        reinterpret_cast<float*>(output)[index] = value;
    }
}

__global__ void nv12_to_nchw_kernel(
    const unsigned char* y_plane,
    const unsigned char* uv_plane,
    int src_width,
    int src_height,
    int y_pitch,
    int uv_pitch,
    void* output,
    int dst_height,
    int dst_width,
    int channels,
    int dtype_code
) {
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_width || dst_y >= dst_height) {
        return;
    }

    const int src_x = scaled_source_index(dst_x, dst_width, src_width);
    const int src_y = scaled_source_index(dst_y, dst_height, src_height);
    const int y_value = static_cast<int>(y_plane[src_y * y_pitch + src_x]);
    const int uv_index = (src_y / 2) * uv_pitch + (src_x / 2) * 2;
    const int u_value = static_cast<int>(uv_plane[uv_index]);
    const int v_value = static_cast<int>(uv_plane[uv_index + 1]);

    const int c = y_value > 16 ? y_value - 16 : 0;
    const int d = u_value - 128;
    const int e = v_value - 128;
    const float b = static_cast<float>(clamp_u8((298 * c + 516 * d + 128) >> 8)) / 255.0f;
    const float g = static_cast<float>(clamp_u8((298 * c - 100 * d - 208 * e + 128) >> 8)) / 255.0f;
    const float r = static_cast<float>(clamp_u8((298 * c + 409 * e + 128) >> 8)) / 255.0f;

    const size_t plane_size = static_cast<size_t>(dst_height) * static_cast<size_t>(dst_width);
    const size_t pixel_index = static_cast<size_t>(dst_y) * static_cast<size_t>(dst_width)
                               + static_cast<size_t>(dst_x);
    if (channels == 1) {
        write_tensor_value(output, dtype_code, pixel_index, static_cast<float>(y_value) / 255.0f);
        return;
    }

    write_tensor_value(output, dtype_code, pixel_index, r);
    write_tensor_value(output, dtype_code, plane_size + pixel_index, g);
    write_tensor_value(output, dtype_code, plane_size * 2 + pixel_index, b);
    if (channels == 4) {
        write_tensor_value(output, dtype_code, plane_size * 3 + pixel_index, 1.0f);
    }
}

std::string cuda_error_string(cudaError_t error) {
    return std::string(cudaGetErrorName(error)) + ": " + cudaGetErrorString(error);
}

std::string cu_error_string(CUresult result) {
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    std::string out = name != nullptr ? name : "CUDA_ERROR_UNKNOWN";
    if (text != nullptr) {
        out += ": ";
        out += text;
    }
    return out;
}

bool ensure_cuda_context(std::string* detail) {
    const CUresult init = cuInit(0);
    if (init != CUDA_SUCCESS) {
        *detail = cu_error_string(init);
        return false;
    }
    const cudaError_t runtime_init = cudaFree(nullptr);
    if (runtime_init != cudaSuccess) {
        *detail = cuda_error_string(runtime_init);
        return false;
    }
    return true;
}

uint64_t register_allocation(void* ptr, uint64_t nbytes) {
    std::lock_guard<std::mutex> lock(g_allocations_mutex);
    const uint64_t token = g_next_release_token++;
    g_allocations[token] = DeviceAllocation{ptr, nbytes};
    return token;
}

bool valid_egl_image(EGLImageKHR image) {
    return image != nullptr && image != EGL_NO_IMAGE_KHR;
}

bool copy_array_frame_to_linear_nv12(
    const CUeglFrame& egl_frame,
    int width,
    int height,
    unsigned char** y_linear,
    unsigned char** uv_linear,
    int* y_pitch,
    int* uv_pitch,
    std::string* detail
) {
    if (egl_frame.planeCount < 2 || egl_frame.frame.pArray[0] == nullptr
        || egl_frame.frame.pArray[1] == nullptr) {
        *detail = "CUDA array EGL frame does not expose two NV12 planes.";
        return false;
    }
    *y_pitch = width;
    *uv_pitch = width;
    const size_t y_bytes = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t uv_bytes = static_cast<size_t>(width) * static_cast<size_t>(height / 2);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(y_linear), y_bytes);
    if (err != cudaSuccess) {
        *detail = "cudaMalloc y_linear failed: " + cuda_error_string(err);
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(uv_linear), uv_bytes);
    if (err != cudaSuccess) {
        cudaFree(*y_linear);
        *y_linear = nullptr;
        *detail = "cudaMalloc uv_linear failed: " + cuda_error_string(err);
        return false;
    }
    err = cudaMemcpy2DFromArrayAsync(
        *y_linear,
        static_cast<size_t>(*y_pitch),
        reinterpret_cast<cudaArray_t>(egl_frame.frame.pArray[0]),
        0,
        0,
        static_cast<size_t>(width),
        static_cast<size_t>(height),
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        cudaFree(*y_linear);
        cudaFree(*uv_linear);
        *y_linear = nullptr;
        *uv_linear = nullptr;
        *detail = "cudaMemcpy2DFromArrayAsync Y failed: " + cuda_error_string(err);
        return false;
    }
    err = cudaMemcpy2DFromArrayAsync(
        *uv_linear,
        static_cast<size_t>(*uv_pitch),
        reinterpret_cast<cudaArray_t>(egl_frame.frame.pArray[1]),
        0,
        0,
        static_cast<size_t>(width),
        static_cast<size_t>(height / 2),
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        cudaFree(*y_linear);
        cudaFree(*uv_linear);
        *y_linear = nullptr;
        *uv_linear = nullptr;
        *detail = "cudaMemcpy2DFromArrayAsync UV failed: " + cuda_error_string(err);
        return false;
    }
    return true;
}

bool resolve_nv12_planes(
    const CUeglFrame& egl_frame,
    int width,
    int height,
    const unsigned char** y_plane,
    const unsigned char** uv_plane,
    unsigned char** y_linear_owner,
    unsigned char** uv_linear_owner,
    int* y_pitch,
    int* uv_pitch,
    std::string* detail
) {
    if (egl_frame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
        if (egl_frame.frame.pPitch[0] == nullptr) {
            *detail = "PITCH EGL frame does not expose Y plane.";
            return false;
        }
        *y_pitch = static_cast<int>(egl_frame.pitch);
        *uv_pitch = static_cast<int>(egl_frame.pitch);
        *y_plane = reinterpret_cast<const unsigned char*>(egl_frame.frame.pPitch[0]);
        if (egl_frame.planeCount > 1 && egl_frame.frame.pPitch[1] != nullptr) {
            *uv_plane = reinterpret_cast<const unsigned char*>(egl_frame.frame.pPitch[1]);
        } else {
            *uv_plane = *y_plane + static_cast<size_t>(*y_pitch) * static_cast<size_t>(height);
        }
        return true;
    }

    if (egl_frame.frameType == CU_EGL_FRAME_TYPE_ARRAY) {
        if (!copy_array_frame_to_linear_nv12(
                egl_frame,
                width,
                height,
                y_linear_owner,
                uv_linear_owner,
                y_pitch,
                uv_pitch,
                detail
            )) {
            return false;
        }
        *y_plane = *y_linear_owner;
        *uv_plane = *uv_linear_owner;
        return true;
    }

    std::ostringstream stream;
    stream << "Unsupported CUeglFrame frameType=" << static_cast<int>(egl_frame.frameType);
    *detail = stream.str();
    return false;
}

bool validate_nv12_plane_layout(
    const unsigned char* y_plane,
    const unsigned char* uv_plane,
    int y_pitch,
    int uv_pitch,
    int width,
    std::string* detail
) {
    if (y_plane == nullptr || uv_plane == nullptr) {
        *detail = "NV12 plane pointer is null.";
        return false;
    }
    if (y_pitch < width || uv_pitch < width) {
        std::ostringstream stream;
        stream << "NV12 pitch is smaller than width: y_pitch=" << y_pitch
               << ", uv_pitch=" << uv_pitch << ", width=" << width << ".";
        *detail = stream.str();
        return false;
    }
    return true;
}

std::string success_json(
    const TensorRequest& request,
    void* device_ptr,
    uint64_t nbytes,
    uint64_t release_token
) {
    std::ostringstream out;
    out << "{\"device_ptr\":" << reinterpret_cast<uintptr_t>(device_ptr)
        << ",\"nbytes\":" << nbytes
        << ",\"shape\":["
        << request.nchw[0] << "," << request.nchw[1] << ","
        << request.nchw[2] << "," << request.nchw[3] << "]"
        << ",\"dtype\":\"" << request.dtype << "\""
        << ",\"backend\":\"novasight_jetson_preprocess_native:jetson_cuda\""
        << ",\"zero_copy\":true"
        << ",\"memory_space\":\"cuda_device\""
        << ",\"release_token\":" << release_token << "}";
    return out.str();
}

}  // namespace

extern "C" uint32_t novasight_abi_version(void) {
    return NOVASIGHT_JETSON_PREPROCESS_ABI_VERSION;
}

extern "C" int novasight_status_json(char* status_json, size_t status_json_size) {
    std::string detail;
    if (!ensure_cuda_context(&detail)) {
        novasight::jetson_preprocess::write_json(
            status_json,
            status_json_size,
            novasight::jetson_preprocess::unavailable_status_json(
                "novasight_jetson_preprocess_native:jetson_cuda",
                "cuda_context_unavailable",
                detail
            )
        );
        return 0;
    }
    novasight::jetson_preprocess::write_json(
        status_json,
        status_json_size,
        "{\"available\":true,"
        "\"ready\":true,"
        "\"backend\":\"novasight_jetson_preprocess_native:jetson_cuda\","
        "\"zero_copy\":true,"
        "\"memory_space\":\"cuda_device\","
        "\"capabilities\":{"
        "\"memory\":[\"dmabuf\",\"nvmm\"],"
        "\"resource_kind\":[\"gstreamer_sample\"],"
        "\"resource_source\":[\"appsink\"],"
        "\"formats\":[\"NV12\"],"
        "\"dtypes\":[\"float32\",\"float16\"]},"
        "\"abi_version\":1}"
    );
    return 0;
}

extern "C" int novasight_prepare_tensor_json(
    const char* payload_json,
    char* result_json,
    size_t result_json_size
) {
    const auto validation = novasight::jetson_preprocess::parse_tensor_request(payload_json);
    if (!validation.valid) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            validation.reason,
            validation.detail
        );
        return 1;
    }
    const TensorRequest& request = validation.request;
    if (request.nchw[0] != 1) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            "unsupported_batch",
            "Jetson CUDA preprocess currently supports N=1."
        );
        return 1;
    }
    if (request.width <= 0 || request.height <= 0 || request.nchw[2] <= 0 || request.nchw[3] <= 0) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            "invalid_geometry",
            "Input and NCHW geometry must be positive."
        );
        return 1;
    }
    if ((request.height % 2) != 0 || (request.width % 2) != 0) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            "unsupported_nv12_geometry",
            "NV12 input width and height must be even."
        );
        return 1;
    }

    std::string detail;
    if (!ensure_cuda_context(&detail)) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            "cuda_context_unavailable",
            detail
        );
        return 1;
    }

    void* surface_buffer = nullptr;
    if (NvBufSurfaceFromFd(request.dmabuf_fd, &surface_buffer) != 0 || surface_buffer == nullptr) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            "nvbufsurface_from_fd_failed",
            "NvBufSurfaceFromFd failed for dmabuf_fd."
        );
        return 1;
    }
    NvBufSurface* surface = static_cast<NvBufSurface*>(surface_buffer);
    if (surface->batchSize < 1 || surface->surfaceList == nullptr) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            "invalid_nvbufsurface",
            "NvBufSurface has no surfaceList entry."
        );
        return 1;
    }
    if (surface->surfaceList[0].colorFormat != NVBUF_COLOR_FORMAT_NV12) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            "unsupported_nvbufsurface_format",
            "NvBufSurface colorFormat is not NVBUF_COLOR_FORMAT_NV12."
        );
        return 1;
    }
    const int surface_width = static_cast<int>(surface->surfaceList[0].width);
    const int surface_height = static_cast<int>(surface->surfaceList[0].height);
    if (surface_width != request.width || surface_height != request.height) {
        std::ostringstream geometry_detail;
        geometry_detail << "NvBufSurface geometry "
                        << surface_width << "x" << surface_height
                        << " does not match request "
                        << request.width << "x" << request.height << ".";
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            "nvbufsurface_geometry_mismatch",
            geometry_detail.str()
        );
        return 1;
    }

    if (NvBufSurfaceMapEglImage(surface, 0) != 0) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            "nvbufsurface_map_egl_failed",
            "NvBufSurfaceMapEglImage failed."
        );
        return 1;
    }

    CUgraphicsResource cuda_resource = nullptr;
    void* output_device = nullptr;
    unsigned char* y_linear_owner = nullptr;
    unsigned char* uv_linear_owner = nullptr;
    int return_code = 1;

    do {
        EGLImageKHR egl_image = surface->surfaceList[0].mappedAddr.eglImage;
        if (!valid_egl_image(egl_image)) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "egl_image_unavailable",
                "NvBufSurface did not produce a valid EGLImage."
            );
            break;
        }

        CUresult cu_result = cuGraphicsEGLRegisterImage(
            &cuda_resource,
            egl_image,
            CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY
        );
        if (cu_result != CUDA_SUCCESS) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "cuda_egl_register_failed",
                cu_error_string(cu_result)
            );
            break;
        }

        CUeglFrame egl_frame{};
        cu_result = cuGraphicsResourceGetMappedEglFrame(&egl_frame, cuda_resource, 0, 0);
        if (cu_result != CUDA_SUCCESS) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "cuda_egl_frame_failed",
                cu_error_string(cu_result)
            );
            break;
        }

        const unsigned char* y_plane = nullptr;
        const unsigned char* uv_plane = nullptr;
        int y_pitch = 0;
        int uv_pitch = 0;
        if (!resolve_nv12_planes(
                egl_frame,
                request.width,
                request.height,
                &y_plane,
                &uv_plane,
                &y_linear_owner,
                &uv_linear_owner,
                &y_pitch,
                &uv_pitch,
                &detail
            )) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "cuda_egl_nv12_plane_unavailable",
                detail
            );
            break;
        }
        if (!validate_nv12_plane_layout(
                y_plane,
                uv_plane,
                y_pitch,
                uv_pitch,
                request.width,
                &detail
            )) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "cuda_egl_nv12_plane_invalid",
                detail
            );
            break;
        }

        const uint64_t nbytes = novasight::jetson_preprocess::tensor_nbytes(request);
        if (nbytes == 0) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "invalid_tensor_nbytes",
                "Tensor nbytes resolved to zero."
            );
            break;
        }
        cudaError_t err = cudaMalloc(&output_device, static_cast<size_t>(nbytes));
        if (err != cudaSuccess) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "cuda_malloc_output_failed",
                cuda_error_string(err)
            );
            output_device = nullptr;
            break;
        }

        const dim3 block(16, 16);
        const dim3 grid(
            (request.nchw[3] + block.x - 1) / block.x,
            (request.nchw[2] + block.y - 1) / block.y
        );
        const int dtype_code = request.dtype == "float16" ? 16 : 32;
        nv12_to_nchw_kernel<<<grid, block>>>(
            y_plane,
            uv_plane,
            request.width,
            request.height,
            y_pitch,
            uv_pitch,
            output_device,
            request.nchw[2],
            request.nchw[3],
            request.nchw[1],
            dtype_code
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "cuda_kernel_launch_failed",
                cuda_error_string(err)
            );
            break;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "cuda_kernel_sync_failed",
                cuda_error_string(err)
            );
            break;
        }

        const uint64_t release_token = register_allocation(output_device, nbytes);
        novasight::jetson_preprocess::write_json(
            result_json,
            result_json_size,
            success_json(request, output_device, nbytes, release_token)
        );
        output_device = nullptr;
        return_code = 0;
    } while (false);

    if (y_linear_owner != nullptr) {
        cudaFree(y_linear_owner);
    }
    if (uv_linear_owner != nullptr) {
        cudaFree(uv_linear_owner);
    }
    if (cuda_resource != nullptr) {
        cuGraphicsUnregisterResource(cuda_resource);
    }
    NvBufSurfaceUnMapEglImage(surface, 0);
    if (output_device != nullptr) {
        cudaFree(output_device);
    }
    return return_code;
}

extern "C" int novasight_release_tensor(uint64_t release_token) {
    DeviceAllocation allocation;
    {
        std::lock_guard<std::mutex> lock(g_allocations_mutex);
        const auto found = g_allocations.find(release_token);
        if (found == g_allocations.end()) {
            return 1;
        }
        allocation = found->second;
        g_allocations.erase(found);
    }
    if (allocation.ptr == nullptr) {
        return 1;
    }
    const cudaError_t err = cudaFree(allocation.ptr);
    return err == cudaSuccess ? 0 : 1;
}
