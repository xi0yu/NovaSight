#include "novasight_gpu_frame.h"
#include "novasight_yolo_gpu.hpp"
#include "novasight_tensorrt_runtime.h"
#include <cudaEGL.h>
#include <nvbufsurface.h>
#include <gst/gst.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

void enqueue_rgba_to_chw(const unsigned char*, unsigned, void*, unsigned, unsigned,
                        bool, bool, float, cudaStream_t);
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void cuda_ok(cudaError_t code, const char* operation) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorName(code)
            + " (" + std::to_string(static_cast<int>(code)) + "): " + cudaGetErrorString(code));
}
void driver_ok(CUresult code, const char* operation) {
    if (code == CUDA_SUCCESS) return;
    const char* name = "CUDA_ERROR_UNKNOWN";
    const char* message = "CUDA EGL operation failed";
    cuGetErrorName(code, &name);
    cuGetErrorString(code, &message);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "unknown")
        + " (" + std::to_string(static_cast<int>(code)) + "): " + (message ? message : "unknown"));
}
void error_text(char* out, size_t size, const char* stage, const char* text) noexcept {
    if (out && size) std::snprintf(out, size, "GPU frame [%s]: %s", stage, text);
}
struct ImageLease {
    GstBuffer* buffer;
    cudaStream_t stream;
    GstMapInfo map{};
    NvBufSurface* surface = nullptr;
    CUgraphicsResource resource = nullptr;
    bool mapped = false, owns_egl = false, producer_done = false;
    ~ImageLease() {
        // Also protects early error exits after submitting image-reading work.
        if (!producer_done) cudaStreamSynchronize(stream);
        if (resource) cuGraphicsUnregisterResource(resource);
        if (owns_egl) NvBufSurfaceUnMapEglImage(surface, 0);
        if (mapped) gst_buffer_unmap(buffer, &map);
    }
    CUeglFrame import(unsigned width, unsigned height) {
        require(GST_IS_BUFFER(buffer) && gst_buffer_map(buffer, &map, GST_MAP_READ), "NVMM descriptor mapping failed");
        mapped = true;
        require(map.size >= sizeof(NvBufSurface), "Missing NVMM surface descriptor");
        surface = reinterpret_cast<NvBufSurface*>(map.data);
        require(surface->memType == NVBUF_MEM_SURFACE_ARRAY && surface->numFilled == 1
            && surface->batchSize >= 1 && surface->surfaceList, "GPU path requires one hardware NVMM surface");
        auto& image = surface->surfaceList[0];
        require(image.width == width && image.height == height && image.colorFormat == NVBUF_COLOR_FORMAT_RGBA,
            "GPU input size or color format differs from admitted model");
        if (!image.mappedAddr.eglImage) {
            require(NvBufSurfaceMapEglImage(surface, 0) == 0, "NVMM EGL mapping failed");
            owns_egl = true;
        }
        driver_ok(cuGraphicsEGLRegisterImage(&resource, image.mappedAddr.eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY),
            "cuGraphicsEGLRegisterImage");
        CUeglFrame frame{};
        driver_ok(cuGraphicsResourceGetMappedEglFrame(&frame, resource, 0, 0), "cuGraphicsResourceGetMappedEglFrame");
        require(frame.frameType == CU_EGL_FRAME_TYPE_PITCH && frame.planeCount == 1 && frame.numChannels == 4
            && frame.cuFormat == CU_AD_FORMAT_UNSIGNED_INT8 && frame.eglColorFormat == CU_EGL_COLOR_FORMAT_ABGR
            && frame.width == width && frame.height == height && frame.pitch >= width * 4 && frame.frame.pPitch[0],
            "Unsupported CUDA image layout; CPU fallback forbidden");
        return frame;
    }
    void release() {
        cuda_ok(cudaStreamSynchronize(stream), "producer completion before NVMM release (asynchronous)");
        producer_done = true;
        driver_ok(cuGraphicsUnregisterResource(resource), "cuGraphicsUnregisterResource"); resource = nullptr;
        if (owns_egl) {
            require(NvBufSurfaceUnMapEglImage(surface, 0) == 0, "NVMM EGL release failed");
            owns_egl = false;
        }
    }
};
}

struct novasight_gpu_frame {
    novasight_gpu_frame_config config{};
    novasight_tensorrt_engine* engine = nullptr;
    novasight_engine_spec spec{};
    void* input = nullptr;
    cudaStream_t producer = nullptr;
    cudaEvent_t ready = nullptr;
    std::unique_ptr<novasight::YoloGpuPostprocessor> post;
    bool failed = false, warmed = false;
    ~novasight_gpu_frame() {
        post.reset();
        if (engine) novasight_tensorrt_destroy(engine);
        if (producer) cudaStreamSynchronize(producer);
        if (ready) cudaEventDestroy(ready);
        if (producer) cudaStreamDestroy(producer);
        if (input) cudaFree(input);
    }
};

extern "C" int novasight_gpu_frame_create(const char* engine, const char* input_name, const char* output_name,
    const novasight_gpu_frame_config* config, novasight_gpu_frame** out, char* error, size_t error_size) {
    if (out) *out = nullptr;
    const char* stage = "contract";
    try {
        require(out && engine && input_name && output_name && config, "Missing GPU frame construction argument");
        const auto& c = *config;
        require(c.abi_version == NOVASIGHT_GPU_FRAME_ABI, "GPU frame ABI mismatch");
        require(c.width && c.height && c.width <= 16384 && c.height <= 16384
            && c.classes && c.classes <= 1024 && c.candidates && c.candidates <= 32768,
            "GPU frame dimensions exceed supported bounds");
        require(c.channels_first <= 1 && c.has_objectness <= 1 && c.bgr <= 1 && c.cuda_graph <= 1
            && (c.input_dtype == 1 || c.input_dtype == 2) && (c.output_dtype == 1 || c.output_dtype == 2),
            "Invalid GPU frame flags or tensor dtype");
        require(std::isfinite(c.scale) && c.scale > 0 && std::isfinite(255.0f * c.scale)
            && (c.input_dtype != 2 || 255.0f * c.scale <= 65504.0f),
            "GPU normalization scale overflows the input dtype or is not positive/finite");
        auto owner = std::make_unique<novasight_gpu_frame>();
        owner->config = c;
        stage = "TensorRT initialization";
        cuda_ok(cudaSetDevice(0), "cudaSetDevice(0)");
        const uint64_t shape[] = {1, 3, c.height, c.width};
        char detail[1024]{};
        require(novasight_tensorrt_create(engine, shape, 4, &owner->engine, &owner->spec, detail, sizeof(detail)) == 0, detail);
        const auto& spec = owner->spec;
        require(spec.input.dtype == c.input_dtype && spec.output_count == 1
            && std::strcmp(spec.input.name, input_name) == 0 && std::strcmp(spec.outputs[0].name, output_name) == 0,
            "Engine bindings differ from admitted manifest");
        const auto& output = spec.outputs[0];
        const unsigned start = output.rank == 3 ? 1 : 0;
        const int64_t channels = c.classes + (c.has_objectness ? 5 : 4);
        require((output.rank == 2 || (output.rank == 3 && output.dimensions[0] == 1))
            && output.dtype == c.output_dtype && output.dimensions[start] == (c.channels_first ? channels : c.candidates)
            && output.dimensions[start + 1] == (c.channels_first ? c.candidates : channels),
            "Engine output shape/dtype differs from raw YOLO contract");
        stage = "GPU workspace initialization";
        cuda_ok(cudaMalloc(&owner->input, spec.input.nbytes), "cudaMalloc input");
        cuda_ok(cudaStreamCreateWithFlags(&owner->producer, cudaStreamNonBlocking), "cudaStreamCreateWithFlags producer");
        cuda_ok(cudaEventCreateWithFlags(&owner->ready, cudaEventDisableTiming), "cudaEventCreateWithFlags input-ready");
        owner->post = std::make_unique<novasight::YoloGpuPostprocessor>(novasight::YoloGpuConfig{
            c.candidates, c.classes, c.width, c.height, bool(c.channels_first), bool(c.has_objectness),
            c.output_dtype == 2, c.confidence_threshold, c.nms_threshold, c.top_k});
        *out = owner.release();
        return 0;
    } catch (const std::exception& e) { error_text(error, error_size, stage, e.what()); return 1; }
    catch (...) { error_text(error, error_size, stage, "non-standard C++ exception"); return 1; }
}

extern "C" int novasight_gpu_frame_process(novasight_gpu_frame* owner, void* buffer,
    novasight_gpu_result* out, novasight_gpu_timings* timings, char* error, size_t error_size) {
    static_assert(sizeof(novasight_gpu_timings) == 2 * sizeof(uint32_t) + 6 * sizeof(uint64_t),
        "novasight_gpu_timings layout must match the Rust #[repr(C)] mirror");
    if (out) std::memset(out, 0, sizeof(*out));
    if (timings) std::memset(timings, 0, sizeof(*timings));
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    auto t1 = t0, t2 = t0, t3 = t0, t4 = t0, t5 = t0;
    const char* stage = "frame arguments";
    try {
        require(owner && buffer && out, "Missing GPU frame execution argument");
        require(!owner->failed, "GPU frame context is invalid after a prior error");
        const novasight::YoloGpuResult* result = nullptr;
        {
            ImageLease lease{static_cast<GstBuffer*>(buffer), owner->producer};
            const auto& c = owner->config;
            stage = "NVMM/EGL import";
            const auto frame = lease.import(c.width, c.height);
            t1 = clock::now();
            stage = "preprocess launch";
            enqueue_rgba_to_chw(static_cast<const unsigned char*>(frame.frame.pPitch[0]), frame.pitch,
                owner->input, c.width, c.height, c.input_dtype == 2, bool(c.bgr), c.scale, owner->producer);
            cuda_ok(cudaGetLastError(), "RGBA-to-NCHW launch");
            cuda_ok(cudaEventRecord(owner->ready, owner->producer), "cudaEventRecord input-ready");
            novasight_device_tensor_view input{};
            input.device_ptr = reinterpret_cast<uintptr_t>(owner->input);
            input.nbytes = owner->spec.input.nbytes; input.rank = 4; input.dtype = c.input_dtype;
            std::copy_n(owner->spec.input.dimensions, 4, input.dimensions);
            novasight_device_tensor_view output[NOVASIGHT_TENSORRT_MAX_OUTPUTS]{};
            uint32_t count = 0; uint64_t stream = 0; char detail[1024]{};
            stage = "TensorRT enqueue";
            require(novasight_tensorrt_enqueue_device(owner->engine, &input,
                reinterpret_cast<uintptr_t>(owner->ready), output, NOVASIGHT_TENSORRT_MAX_OUTPUTS,
                &count, &stream, detail, sizeof(detail)) == 0, detail);
            require(count == 1, "Unexpected GPU tensor count");
            stage = "YOLO decode/sort/NMS enqueue";
            owner->post->enqueue(reinterpret_cast<void*>(output[0].device_ptr), output[0].nbytes,
                reinterpret_cast<cudaStream_t>(stream));
            t2 = clock::now();
            // A wait can observe an error from any earlier producer/kernel.
            // Report the observation point, not an invented failing kernel.
            stage = "GPU completion (asynchronous upstream work)";
            result = &owner->post->wait();
            require(novasight_tensorrt_finish_device(owner->engine, detail, sizeof(detail)) == 0, detail);
            t3 = clock::now();
            stage = "NVMM/EGL release";
            lease.release();
        }  // ImageLease destructor unmaps the GstBuffer descriptor.
        t4 = clock::now();
        if (owner->config.cuda_graph && !owner->warmed) {
            stage = "TensorRT graph capture";
            char graph_detail[1024]{};
            require(novasight_tensorrt_capture_device_graph(owner->engine, graph_detail, sizeof(graph_detail)) == 0, graph_detail);
        }
        owner->warmed = true;
        t5 = clock::now();
        auto ns = [](clock::time_point a, clock::time_point b) -> uint64_t {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
        };
        if (timings) {
            timings->abi_version = NOVASIGHT_GPU_TIMINGS_ABI;
            timings->warmed = owner->warmed ? 1u : 0u;
            timings->import_ns = ns(t0, t1);
            timings->launch_ns = ns(t1, t2);
            timings->gpu_wait_ns = ns(t2, t3);
            timings->release_ns = ns(t3, t4);
            timings->graph_capture_ns = ns(t4, t5);
            timings->total_ns = ns(t0, t5);
        }
        static_assert(sizeof(novasight_gpu_result) == sizeof(novasight::YoloGpuResult));
        static_assert(sizeof(novasight_gpu_detection) == sizeof(novasight::YoloGpuDetection));
        std::memcpy(out, result, sizeof(*out));
        return 0;
    } catch (const std::exception& e) {
        if (owner) owner->failed = true;
        error_text(error, error_size, stage, e.what()); return 1;
    } catch (...) {
        if (owner) owner->failed = true;
        error_text(error, error_size, stage, "non-standard C++ exception"); return 1;
    }
}
extern "C" void novasight_gpu_frame_destroy(novasight_gpu_frame* context) { delete context; }
