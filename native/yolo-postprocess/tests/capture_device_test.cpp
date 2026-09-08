// Bounded, output-free experiment for the measured 1080p120 / 320px ROI fixture.
// This is not a production adapter: no model activation, preview or control.
#include "novasight_yolo_gpu.hpp"
#include "novasight_tensorrt_runtime.h"
#include <cudaEGL.h>
#include <nvbufsurface.h>
#include <gst/app/gstappsink.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

void enqueue_rgba_to_rgb_chw(const unsigned char*, unsigned, float*, cudaStream_t);

namespace {
constexpr unsigned side = 256, pixels = side * side;
void require(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
void cuda_ok(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
void driver_ok(CUresult status) {
    if (status == CUDA_SUCCESS) return;
    const char* error = "CUDA driver failure";
    cuGetErrorString(status, &error);
    throw std::runtime_error(error);
}
std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Runtime {
    novasight_tensorrt_engine* engine = nullptr;
    void* input = nullptr;
    cudaStream_t producer = nullptr;
    cudaEvent_t ready = nullptr;
    GstElement* pipeline = nullptr;
    GstElement* sink = nullptr;
    ~Runtime() {
        // Borrowed engine work must finish before the persistent tensor is freed.
        if (engine) novasight_tensorrt_destroy(engine);
        if (producer) cudaStreamSynchronize(producer);
        if (pipeline) gst_element_set_state(pipeline, GST_STATE_NULL);
        if (sink) gst_object_unref(sink);
        if (pipeline) gst_object_unref(pipeline);
        if (ready) cudaEventDestroy(ready);
        if (producer) cudaStreamDestroy(producer);
        if (input) cudaFree(input);
    }
};

struct Frame {
    GstSample* sample;
    cudaStream_t producer;
    GstMapInfo map{};
    NvBufSurface* surface = nullptr;
    CUgraphicsResource resource = nullptr;
    bool mapped = false, owns_egl = false;
    Frame(GstSample* value, cudaStream_t stream) : sample(value), producer(stream) {}
    ~Frame() {
        // Covers errors after the image-reading kernel was submitted.
        cudaStreamSynchronize(producer);
        if (resource) cuGraphicsUnregisterResource(resource);
        if (owns_egl) NvBufSurfaceUnMapEglImage(surface, 0);
        if (mapped) gst_buffer_unmap(gst_sample_get_buffer(sample), &map);
        gst_sample_unref(sample);
    }
    CUeglFrame import() {
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        require(buffer && gst_buffer_map(buffer, &map, GST_MAP_READ), "NVMM descriptor mapping failed");
        mapped = true;
        require(map.size >= sizeof(NvBufSurface), "Missing NvBufSurface descriptor");
        surface = reinterpret_cast<NvBufSurface*>(map.data);
        require(surface->memType == NVBUF_MEM_SURFACE_ARRAY && surface->numFilled == 1
            && surface->surfaceList, "Expected one Jetson hardware surface");
        auto& image = surface->surfaceList[0];
        require(image.width == side && image.height == side && image.colorFormat == NVBUF_COLOR_FORMAT_RGBA,
            "Expected 256px RGBA hardware surface");
        if (!image.mappedAddr.eglImage) {
            require(NvBufSurfaceMapEglImage(surface, 0) == 0, "NVMM EGL mapping failed");
            owns_egl = true;
        }
        driver_ok(cuGraphicsEGLRegisterImage(&resource, image.mappedAddr.eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY));
        CUeglFrame frame{};
        driver_ok(cuGraphicsResourceGetMappedEglFrame(&frame, resource, 0, 0));
        // CUDA EGL ABGR names the packed word; its byte ordering is RGBA.
        require(frame.frameType == CU_EGL_FRAME_TYPE_PITCH && frame.planeCount == 1
            && frame.numChannels == 4 && frame.cuFormat == CU_AD_FORMAT_UNSIGNED_INT8
            && frame.eglColorFormat == CU_EGL_COLOR_FORMAT_ABGR && frame.width == side
            && frame.height == side && frame.pitch >= side * 4 && frame.frame.pPitch[0],
            "Unsupported CUDA EGL image contract; no CPU fallback");
        return frame;
    }
    void release() {
        driver_ok(cuGraphicsUnregisterResource(resource));
        resource = nullptr;
        if (owns_egl) {
            require(NvBufSurfaceUnMapEglImage(surface, 0) == 0, "NVMM EGL release failed");
            owns_egl = false;
        }
    }
};

void self_test() {
    constexpr unsigned pitch = side * 4 + 64;
    std::vector<unsigned char> rgba(pitch * side, 237);
    for (unsigned y = 0; y < side; ++y) for (unsigned x = 0; x < side; ++x) {
        auto* p = rgba.data() + y * pitch + x * 4;
        p[0] = x; p[1] = y; p[2] = (x + y) % 256; p[3] = 255;
    }
    Runtime fixture;
    cuda_ok(cudaStreamCreateWithFlags(&fixture.producer, cudaStreamNonBlocking));
    cuda_ok(cudaMalloc(&fixture.input, pixels * 3 * sizeof(float)));
    void* device_rgba = nullptr;
    cuda_ok(cudaMalloc(&device_rgba, rgba.size()));
    try {
        cuda_ok(cudaMemcpy(device_rgba, rgba.data(), rgba.size(), cudaMemcpyHostToDevice));
        enqueue_rgba_to_rgb_chw(static_cast<const unsigned char*>(device_rgba), pitch,
            static_cast<float*>(fixture.input), fixture.producer);
        cuda_ok(cudaGetLastError());
        cuda_ok(cudaStreamSynchronize(fixture.producer));
        std::vector<float> actual(pixels * 3);
        cuda_ok(cudaMemcpy(actual.data(), fixture.input, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (unsigned y = 0; y < side; ++y) for (unsigned x = 0; x < side; ++x)
            for (unsigned c = 0; c < 3; ++c)
                require(std::fabs(actual[c * pixels + y * side + x]
                    - float(rgba[y * pitch + x * 4 + c]) / 255.0f) <= 1e-7f,
                    "RGBA stride / channel / normalization mismatch");
        cudaFree(device_rgba);
    } catch (...) {
        cudaStreamSynchronize(fixture.producer);
        cudaFree(device_rgba);
        throw;
    }
    std::cout << "CAPTURE_PREPROCESS_PASS pixels=65536 channels=3 padded_stride=true\n";
}

void run(const char* engine_path, unsigned seconds) {
    Runtime runtime;
    novasight_engine_spec spec{};
    char error[1024]{};
    auto trt_ok = [&](int status) { require(status == 0, error); };
    trt_ok(novasight_tensorrt_create(engine_path, nullptr, 0, &runtime.engine, &spec, error, sizeof(error)));
    require(spec.input.rank == 4 && spec.input.dtype == NOVASIGHT_TENSOR_DTYPE_FLOAT32
        && spec.input.dimensions[0] == 1 && spec.input.dimensions[1] == 3
        && spec.input.dimensions[2] == side && spec.input.dimensions[3] == side
        && spec.input.nbytes == pixels * 3 * sizeof(float) && spec.output_count == 1
        && spec.outputs[0].rank == 3 && spec.outputs[0].dtype == NOVASIGHT_TENSOR_DTYPE_FLOAT32
        && spec.outputs[0].dimensions[0] == 1 && spec.outputs[0].dimensions[1] == 9
        && spec.outputs[0].dimensions[2] == 1344, "Engine differs from the explicit capture fixture");
    cuda_ok(cudaMalloc(&runtime.input, spec.input.nbytes));
    cuda_ok(cudaStreamCreateWithFlags(&runtime.producer, cudaStreamNonBlocking));
    cuda_ok(cudaEventCreateWithFlags(&runtime.ready, cudaEventDisableTiming));
    novasight::YoloGpuPostprocessor post({1344, 5, side, side, true, false, false, .65f, .45f, 256});
    novasight_device_tensor_view input{};
    input.device_ptr = reinterpret_cast<std::uintptr_t>(runtime.input);
    input.nbytes = spec.input.nbytes; input.rank = spec.input.rank; input.dtype = spec.input.dtype;
    for (unsigned d = 0; d < input.rank; ++d) input.dimensions[d] = spec.input.dimensions[d];
    const auto ready = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(runtime.ready));

    gst_init(nullptr, nullptr);
    GError* parse_error = nullptr;
    runtime.pipeline = gst_parse_launch(
        "v4l2src name=capture-source device=/dev/video0 io-mode=2 do-timestamp=true ! "
        "image/jpeg,width=1920,height=1080,framerate=120/1 ! "
        "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        "jpegparse ! nvv4l2decoder mjpeg=1 ! video/x-raw(memory:NVMM),format=I420 ! "
        "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        "nvvidconv left=800 right=1120 top=380 bottom=700 ! "
        "video/x-raw(memory:NVMM),format=RGBA,width=256,height=256,pixel-aspect-ratio=1/1 ! "
        "appsink name=gpu-input max-buffers=1 drop=true sync=false wait-on-eos=false", &parse_error);
    if (parse_error) {
        std::string message = parse_error->message;
        g_error_free(parse_error);
        throw std::runtime_error(message);
    }
    require(runtime.pipeline, "Capture pipeline construction failed");
    runtime.sink = gst_bin_get_by_name(GST_BIN(runtime.pipeline), "gpu-input");
    require(runtime.sink && GST_IS_APP_SINK(runtime.sink), "Capture appsink missing");
    require(gst_element_set_state(runtime.pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
        "Capture failed to enter PLAYING");
    using Span = void (*)(const char*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);
    auto span = reinterpret_cast<Span>(dlsym(RTLD_DEFAULT, "novasight_p0_span"));
    std::vector<double> latency;
    latency.reserve(120 * seconds + 120);
    std::uint64_t first = 0, last_pts = GST_CLOCK_TIME_NONE, last = 0;
    unsigned frames = 0, nonempty = 0, truncated = 0;
    for (;;) {
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(runtime.sink), GST_SECOND);
        require(sample, "No hardware frame within one second; stopping without fallback");
        Frame lease(sample, runtime.producer);
        const auto begin = now_ns();
        if (!first) first = begin;
        const auto pts = GST_BUFFER_PTS(gst_sample_get_buffer(sample));
        require(pts != GST_CLOCK_TIME_NONE && (last_pts == GST_CLOCK_TIME_NONE || pts > last_pts),
            "Missing or nonmonotonic capture frame identity");
        last_pts = pts;
        const auto frame = lease.import();
        enqueue_rgba_to_rgb_chw(static_cast<const unsigned char*>(frame.frame.pPitch[0]), frame.pitch,
            static_cast<float*>(runtime.input), runtime.producer);
        cuda_ok(cudaGetLastError());
        cuda_ok(cudaEventRecord(runtime.ready, runtime.producer));
        novasight_device_tensor_view output[NOVASIGHT_TENSORRT_MAX_OUTPUTS]{};
        unsigned count = 0;
        std::uint64_t stream = 0;
        trt_ok(novasight_tensorrt_enqueue_device(runtime.engine, &input, ready, output,
            NOVASIGHT_TENSORRT_MAX_OUTPUTS, &count, &stream, error, sizeof(error)));
        require(count == 1, "Unexpected device output count");
        post.enqueue(reinterpret_cast<void*>(output[0].device_ptr), output[0].nbytes,
            reinterpret_cast<cudaStream_t>(stream));
        const auto& result = post.wait();
        trt_ok(novasight_tensorrt_finish_device(runtime.engine, error, sizeof(error)));
        const auto end = now_ns();
        if (span) span("gpu_candidate/result", begin, end, pts, result.count);
        lease.release();
        last = end;
        ++frames;
        if (begin - first >= 2000000000ULL) {
            latency.push_back(double(end - begin) / 1000.0);
            nonempty += result.count != 0;
            truncated += result.truncated;
        }
        if (end - first >= (seconds + 2ULL) * 1000000000ULL) break;
    }
    require(!latency.empty(), "No steady-state capture samples");
    require(gst_element_set_state(runtime.pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
        "Capture failed to stop");
    std::sort(latency.begin(), latency.end());
    auto percentile = [&](double p) { return latency[std::size_t(std::ceil(latency.size() * p)) - 1]; };
    std::cout << "{\"capture_gpu_pass\":true,\"physical_output\":false,\"frames\":" << frames
        << ",\"steady_samples\":" << latency.size() << ",\"nonempty_frames\":" << nonempty
        << ",\"truncated\":" << truncated << ",\"elapsed_ns\":" << last - first
        << ",\"image_ready_to_result_p50_us\":" << percentile(.5)
        << ",\"image_ready_to_result_p95_us\":" << percentile(.95)
        << ",\"image_ready_to_result_p99_us\":" << percentile(.99)
        << ",\"final_d2h_bytes\":" << sizeof(novasight::YoloGpuResult) << "}\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) self_test();
        else {
            require(argc == 3, "Usage: capture_device_test ENGINE SECONDS(1..30) | --self-test");
            std::size_t used = 0;
            const auto seconds = std::stoul(argv[2], &used);
            require(used == std::strlen(argv[2]) && seconds >= 1 && seconds <= 30, "Invalid capture duration");
            run(argv[1], static_cast<unsigned>(seconds));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CAPTURE_GPU_FAIL: " << error.what() << '\n';
        return 1;
    }
}
