#include "novasight_jetson_preprocess_native.h"
#include "novasight_jetson_preprocess_native_jetson_support.h"

#include <EGL/egl.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gst/gst.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

using novasight::jetson_preprocess::TensorRequest;
using Clock = std::chrono::steady_clock;

struct DeviceAllocation {
    void* ptr = nullptr;
    uint64_t nbytes = 0;
};

std::mutex g_allocations_mutex;
std::unordered_map<uint64_t, DeviceAllocation> g_allocations;
uint64_t g_next_release_token = 1;

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

__global__ void rgba_to_nchw_kernel(
    const unsigned char* rgba_plane,
    int src_width,
    int src_height,
    int rgba_pitch,
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
    const unsigned char* pixel = rgba_plane
                                 + static_cast<size_t>(src_y) * static_cast<size_t>(rgba_pitch)
                                 + static_cast<size_t>(src_x) * 4;
    const float r = static_cast<float>(pixel[0]) / 255.0f;
    const float g = static_cast<float>(pixel[1]) / 255.0f;
    const float b = static_cast<float>(pixel[2]) / 255.0f;
    const float a = static_cast<float>(pixel[3]) / 255.0f;

    const size_t plane_size = static_cast<size_t>(dst_height) * static_cast<size_t>(dst_width);
    const size_t pixel_index = static_cast<size_t>(dst_y) * static_cast<size_t>(dst_width)
                               + static_cast<size_t>(dst_x);
    if (channels == 1) {
        write_tensor_value(output, dtype_code, pixel_index, (r + g + b) / 3.0f);
        return;
    }

    write_tensor_value(output, dtype_code, pixel_index, r);
    write_tensor_value(output, dtype_code, plane_size + pixel_index, g);
    write_tensor_value(output, dtype_code, plane_size * 2 + pixel_index, b);
    if (channels == 4) {
        write_tensor_value(output, dtype_code, plane_size * 3 + pixel_index, a);
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

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
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

std::string transform_error_string(NvBufSurfTransform_Error error) {
    std::ostringstream stream;
    stream << "NvBufSurfTransform error=" << static_cast<int>(error);
    return stream.str();
}

bool copy_array_frame_to_linear_rgba(
    const CUeglFrame& egl_frame,
    int width,
    int height,
    unsigned char** rgba_linear,
    int* rgba_pitch,
    std::string* detail
) {
    if (egl_frame.planeCount < 1 || egl_frame.frame.pArray[0] == nullptr) {
        *detail = "CUDA array EGL frame does not expose an RGBA plane.";
        return false;
    }
    *rgba_pitch = width * 4;
    const size_t rgba_bytes = static_cast<size_t>(*rgba_pitch) * static_cast<size_t>(height);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(rgba_linear), rgba_bytes);
    if (err != cudaSuccess) {
        *detail = "cudaMalloc rgba_linear failed: " + cuda_error_string(err);
        return false;
    }
    err = cudaMemcpy2DFromArrayAsync(
        *rgba_linear,
        static_cast<size_t>(*rgba_pitch),
        reinterpret_cast<cudaArray_t>(egl_frame.frame.pArray[0]),
        0,
        0,
        static_cast<size_t>(width) * 4,
        static_cast<size_t>(height),
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        cudaFree(*rgba_linear);
        *rgba_linear = nullptr;
        *detail = "cudaMemcpy2DFromArrayAsync RGBA failed: " + cuda_error_string(err);
        return false;
    }
    return true;
}

bool resolve_rgba_plane(
    const CUeglFrame& egl_frame,
    int width,
    int height,
    const unsigned char** rgba_plane,
    unsigned char** rgba_linear_owner,
    int* rgba_pitch,
    std::string* detail
) {
    if (egl_frame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
        if (egl_frame.frame.pPitch[0] == nullptr) {
            *detail = "PITCH EGL frame does not expose RGBA plane.";
            return false;
        }
        *rgba_pitch = static_cast<int>(egl_frame.pitch);
        *rgba_plane = reinterpret_cast<const unsigned char*>(egl_frame.frame.pPitch[0]);
        return true;
    }

    if (egl_frame.frameType == CU_EGL_FRAME_TYPE_ARRAY) {
        if (!copy_array_frame_to_linear_rgba(
                egl_frame,
                width,
                height,
                rgba_linear_owner,
                rgba_pitch,
                detail
            )) {
            return false;
        }
        *rgba_plane = *rgba_linear_owner;
        return true;
    }

    std::ostringstream stream;
    stream << "Unsupported CUeglFrame frameType=" << static_cast<int>(egl_frame.frameType);
    *detail = stream.str();
    return false;
}

bool validate_rgba_plane_layout(
    const unsigned char* rgba_plane,
    int rgba_pitch,
    int width,
    std::string* detail
) {
    if (rgba_plane == nullptr) {
        *detail = "RGBA plane pointer is null.";
        return false;
    }
    if (rgba_pitch < width * 4) {
        std::ostringstream stream;
        stream << "RGBA pitch is smaller than width*4: rgba_pitch=" << rgba_pitch
               << ", width=" << width << ".";
        *detail = stream.str();
        return false;
    }
    return true;
}

bool resolve_source_surface_from_request(
    const TensorRequest& request,
    NvBufSurface** surface,
    GstBuffer** gst_buffer_owner,
    GstMapInfo* gst_map_info,
    bool* gst_buffer_mapped,
    std::string* reason,
    std::string* detail
) {
    *surface = nullptr;
    *gst_buffer_owner = nullptr;
    *gst_buffer_mapped = false;

    if (request.dmabuf_fd >= 0) {
        void* surface_buffer = nullptr;
        if (NvBufSurfaceFromFd(request.dmabuf_fd, &surface_buffer) != 0
            || surface_buffer == nullptr) {
            *reason = "nvbufsurface_from_fd_failed";
            *detail = "NvBufSurfaceFromFd failed for dmabuf_fd.";
            return false;
        }
        *surface = static_cast<NvBufSurface*>(surface_buffer);
        return true;
    }

    if (request.gst_buffer_ptr == 0) {
        *reason = "frame_resource_handle_required";
        *detail = "Native Jetson preprocessing requires a valid dmabuf_fd or gst_buffer_ptr.";
        return false;
    }

    GstBuffer* buffer =
        reinterpret_cast<GstBuffer*>(static_cast<uintptr_t>(request.gst_buffer_ptr));
    if (buffer == nullptr) {
        *reason = "gst_buffer_ptr_invalid";
        *detail = "gst_buffer_ptr resolved to a null GstBuffer.";
        return false;
    }

    *gst_buffer_owner = gst_buffer_ref(buffer);
    if (*gst_buffer_owner == nullptr) {
        *reason = "gst_buffer_ref_failed";
        *detail = "gst_buffer_ref returned null for gst_buffer_ptr.";
        return false;
    }

    if (!gst_buffer_map(*gst_buffer_owner, gst_map_info, GST_MAP_READ)) {
        gst_buffer_unref(*gst_buffer_owner);
        *gst_buffer_owner = nullptr;
        *reason = "gst_buffer_map_nvbufsurface_failed";
        *detail = "gst_buffer_map failed for gst_buffer_ptr NVMM resource.";
        return false;
    }
    *gst_buffer_mapped = true;

    if (gst_map_info->data == nullptr || gst_map_info->size < sizeof(NvBufSurface)) {
        gst_buffer_unmap(*gst_buffer_owner, gst_map_info);
        gst_buffer_unref(*gst_buffer_owner);
        *gst_buffer_owner = nullptr;
        *gst_buffer_mapped = false;
        *reason = "gst_buffer_nvbufsurface_unavailable";
        *detail = "gst_buffer_map did not expose a valid NvBufSurface payload.";
        return false;
    }

    *surface = reinterpret_cast<NvBufSurface*>(gst_map_info->data);
    return true;
}

std::string success_json(
    const TensorRequest& request,
    void* device_ptr,
    uint64_t nbytes,
    uint64_t release_token,
    double nvbufsurftransform_ms,
    double egl_cuda_map_ms,
    double cuda_kernel_ms,
    double total_ms
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
        << ",\"release_token\":" << release_token
        << ",\"timings\":{"
        << "\"nvbufsurftransform_ms\":" << nvbufsurftransform_ms
        << ",\"egl_cuda_map_ms\":" << egl_cuda_map_ms
        << ",\"cuda_kernel_ms\":" << cuda_kernel_ms
        << ",\"total_ms\":" << total_ms
        << "}}";
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
    const auto total_start = Clock::now();
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

    NvBufSurface* surface = nullptr;
    GstBuffer* gst_buffer_owner = nullptr;
    GstMapInfo gst_map_info{};
    bool gst_buffer_mapped = false;
    std::string source_reason;
    std::string source_detail;
    auto cleanup_source_surface = [&]() {
        if (gst_buffer_mapped && gst_buffer_owner != nullptr) {
            gst_buffer_unmap(gst_buffer_owner, &gst_map_info);
            gst_buffer_mapped = false;
        }
        if (gst_buffer_owner != nullptr) {
            gst_buffer_unref(gst_buffer_owner);
            gst_buffer_owner = nullptr;
        }
    };
    if (!resolve_source_surface_from_request(
            request,
            &surface,
            &gst_buffer_owner,
            &gst_map_info,
            &gst_buffer_mapped,
            &source_reason,
            &source_detail
        )) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            source_reason,
            source_detail
        );
        return 1;
    }
    if (surface->batchSize < 1 || surface->surfaceList == nullptr) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            "invalid_nvbufsurface",
            "NvBufSurface has no surfaceList entry."
        );
        cleanup_source_surface();
        return 1;
    }
    if (surface->surfaceList[0].colorFormat != NVBUF_COLOR_FORMAT_NV12) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            "unsupported_nvbufsurface_format",
            "NvBufSurface colorFormat is not NVBUF_COLOR_FORMAT_NV12."
        );
        cleanup_source_surface();
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
        cleanup_source_surface();
        return 1;
    }

    NvBufSurface* rgba_surface = nullptr;
    NvBufSurfaceCreateParams create_params{};
    create_params.gpuId = surface->gpuId;
    create_params.width = static_cast<uint32_t>(request.nchw[3]);
    create_params.height = static_cast<uint32_t>(request.nchw[2]);
    create_params.size = 0;
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    create_params.layout = NVBUF_LAYOUT_PITCH;
    create_params.memType = NVBUF_MEM_DEFAULT;
    if (NvBufSurfaceCreate(&rgba_surface, 1, &create_params) != 0 || rgba_surface == nullptr) {
        novasight::jetson_preprocess::write_error_json(
            result_json,
            result_json_size,
            "nvbufsurface_create_rgba_failed",
            "NvBufSurfaceCreate failed for intermediate RGBA tensor source."
        );
        cleanup_source_surface();
        return 1;
    }

    CUgraphicsResource cuda_resource = nullptr;
    void* output_device = nullptr;
    unsigned char* rgba_linear_owner = nullptr;
    bool rgba_egl_mapped = false;
    double nvbufsurftransform_ms = 0.0;
    double egl_cuda_map_ms = 0.0;
    double cuda_kernel_ms = 0.0;
    int return_code = 1;

    do {
        NvBufSurfTransformRect src_rect{};
        src_rect.top = 0;
        src_rect.left = 0;
        src_rect.width = static_cast<uint32_t>(request.width);
        src_rect.height = static_cast<uint32_t>(request.height);
        NvBufSurfTransformRect dst_rect{};
        dst_rect.top = 0;
        dst_rect.left = 0;
        dst_rect.width = static_cast<uint32_t>(request.nchw[3]);
        dst_rect.height = static_cast<uint32_t>(request.nchw[2]);
        NvBufSurfTransformParams transform_params{};
        transform_params.src_rect = &src_rect;
        transform_params.dst_rect = &dst_rect;
        transform_params.transform_flag = NVBUFSURF_TRANSFORM_CROP_SRC
                                          | NVBUFSURF_TRANSFORM_CROP_DST
                                          | NVBUFSURF_TRANSFORM_FILTER;
        transform_params.transform_filter = NvBufSurfTransformInter_Default;

        const auto transform_start = Clock::now();
        const NvBufSurfTransform_Error transform_error =
            NvBufSurfTransform(surface, rgba_surface, &transform_params);
        const auto transform_done = Clock::now();
        nvbufsurftransform_ms = elapsed_ms(transform_start, transform_done);
        if (transform_error != NvBufSurfTransformError_Success) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "nvbufsurftransform_nv12_to_rgba_failed",
                transform_error_string(transform_error)
            );
            break;
        }
        const auto map_start = Clock::now();
        if (NvBufSurfaceMapEglImage(rgba_surface, 0) != 0) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "nvbufsurface_rgba_map_egl_failed",
                "NvBufSurfaceMapEglImage failed for transformed RGBA surface."
            );
            break;
        }
        rgba_egl_mapped = true;

        EGLImageKHR egl_image = rgba_surface->surfaceList[0].mappedAddr.eglImage;
        if (!valid_egl_image(egl_image)) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "egl_image_unavailable",
                "Transformed RGBA NvBufSurface did not produce a valid EGLImage."
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

        const unsigned char* rgba_plane = nullptr;
        int rgba_pitch = 0;
        if (!resolve_rgba_plane(
                egl_frame,
                request.nchw[3],
                request.nchw[2],
                &rgba_plane,
                &rgba_linear_owner,
                &rgba_pitch,
                &detail
            )) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "cuda_egl_rgba_plane_unavailable",
                detail
            );
            break;
        }
        if (!validate_rgba_plane_layout(
                rgba_plane,
                rgba_pitch,
                request.nchw[3],
                &detail
            )) {
            novasight::jetson_preprocess::write_error_json(
                result_json,
                result_json_size,
                "cuda_egl_rgba_plane_invalid",
                detail
            );
            break;
        }
        const auto map_done = Clock::now();
        egl_cuda_map_ms = elapsed_ms(map_start, map_done);

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
        const auto kernel_start = Clock::now();
        rgba_to_nchw_kernel<<<grid, block>>>(
            rgba_plane,
            request.nchw[3],
            request.nchw[2],
            rgba_pitch,
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
        const auto kernel_done = Clock::now();
        cuda_kernel_ms = elapsed_ms(kernel_start, kernel_done);
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
            success_json(
                request,
                output_device,
                nbytes,
                release_token,
                nvbufsurftransform_ms,
                egl_cuda_map_ms,
                cuda_kernel_ms,
                elapsed_ms(total_start, Clock::now())
            )
        );
        output_device = nullptr;
        return_code = 0;
    } while (false);

    if (rgba_linear_owner != nullptr) {
        cudaFree(rgba_linear_owner);
    }
    if (cuda_resource != nullptr) {
        cuGraphicsUnregisterResource(cuda_resource);
    }
    if (rgba_egl_mapped) {
        NvBufSurfaceUnMapEglImage(rgba_surface, 0);
    }
    NvBufSurfaceDestroy(rgba_surface);
    if (output_device != nullptr) {
        cudaFree(output_device);
    }
    cleanup_source_surface();
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
