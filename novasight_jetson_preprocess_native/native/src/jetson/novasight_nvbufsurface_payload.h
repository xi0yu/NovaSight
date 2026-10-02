#ifndef NOVASIGHT_NVBUFSURFACE_PAYLOAD_H
#define NOVASIGHT_NVBUFSURFACE_PAYLOAD_H

#include <cstddef>

namespace novasight::jetson_preprocess {

// GstMapInfo::data is the mapped NvBufSurface object itself in the DeepStream
// NVMM contract. Never dereference its first word as though the payload stored
// another pointer.
inline void* checked_nvbufsurface_payload(
    void* data,
    std::size_t mapped_size,
    std::size_t minimum_size
) noexcept {
    if (data == nullptr || mapped_size < minimum_size) {
        return nullptr;
    }
    return data;
}

}  // namespace novasight::jetson_preprocess

#endif
