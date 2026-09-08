#include <cuda_runtime.h>

// Explicit 256px RGB / 1/255 contract of the isolated capture fixture.
__global__ void rgba_to_rgb_chw(const unsigned char* rgba, unsigned pitch, float* tensor) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const auto* p = rgba + (i / 256) * pitch + (i % 256) * 4;
    for (unsigned c = 0; c < 3; ++c) tensor[c * 256 * 256 + i] = float(p[c]) * (1.0f / 255.0f);
}

void enqueue_rgba_to_rgb_chw(const unsigned char* rgba, unsigned pitch, float* tensor, cudaStream_t stream) {
    rgba_to_rgb_chw<<<256, 256, 0, stream>>>(rgba, pitch, tensor);
}
