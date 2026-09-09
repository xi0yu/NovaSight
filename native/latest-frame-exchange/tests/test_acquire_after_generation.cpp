#include "latest_frame_exchange.hpp"

#include <cassert>

using novasight::exchange::FrameDescriptor;
using novasight::exchange::FrameContent;
using novasight::exchange::FrameMemory;
using novasight::exchange::LatestFrameExchange;

namespace {

FrameDescriptor frame(std::uint64_t generation) {
    return FrameDescriptor{
        generation,
        generation,
        generation,
        1'000 + generation,
        0,
        320,
        320,
        320,
        "NV12",
        "monotonic",
        FrameContent::VideoSurface,
        {nullptr, "GstBuffer", FrameMemory::Nvmm, {}},
    };
}

}  // namespace

int main() {
    LatestFrameExchange exchange;
    exchange.publish(frame(7));

    assert(!exchange.acquire_latest_after(7));
    assert(!exchange.acquire_latest_after(8));

    auto selected = exchange.acquire_latest_after(6);
    assert(selected);
    assert(selected->descriptor().generation == 7);
    assert(!exchange.acquire_latest_after(6));
    return 0;
}
