#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void rgba_to_chw(const unsigned char* rgba, unsigned pitch, T* tensor,
                          unsigned width, unsigned height, bool bgr, float scale) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;
    const auto* pixel = rgba + (i / width) * pitch + (i % width) * 4;
    for (unsigned c = 0; c < 3; ++c)
        tensor[c * width * height + i] = T(float(pixel[bgr ? 2 - c : c]) * scale);
}

void enqueue_rgba_to_chw(const unsigned char* rgba, unsigned pitch, void* tensor,
                        unsigned width, unsigned height, bool half, bool bgr, float scale, cudaStream_t stream) {
    const unsigned blocks = (width * height + 255) / 256;
    if (half) rgba_to_chw<<<blocks, 256, 0, stream>>>(rgba, pitch, static_cast<__half*>(tensor), width, height, bgr, scale);
    else rgba_to_chw<<<blocks, 256, 0, stream>>>(rgba, pitch, static_cast<float*>(tensor), width, height, bgr, scale);
}

// Existing capture fixture exercises the same production preprocessing kernel.
void enqueue_rgba_to_rgb_chw(const unsigned char* rgba, unsigned pitch, float* tensor, cudaStream_t stream) {
    enqueue_rgba_to_chw(rgba, pitch, tensor, 256, 256, false, false, 1.0f / 255.0f, stream);
}
