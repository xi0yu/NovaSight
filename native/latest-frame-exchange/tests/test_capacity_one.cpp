#include "latest_frame_exchange.hpp"

#include <cassert>

using novasight::exchange::FrameDescriptor;
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
        1920,
        1080,
        1920,
        "NV12",
        "monotonic",
        {nullptr, "GstBuffer", FrameMemory::Nvmm, {}},
    };
}

}  // namespace

int main() {
    LatestFrameExchange exchange;
    for (std::uint64_t generation = 1; generation <= 100; ++generation) {
        exchange.publish(frame(generation));
        auto status = exchange.status();
        assert(status.pending_depth == 1);
        assert(status.max_pending_depth == 1);
    }

    auto selected = exchange.acquire_latest_after(0);
    assert(selected);
    assert(selected->descriptor().generation == 100);
    auto status = exchange.status();
    assert(status.pending_depth == 0);
    assert(status.replaced_total == 99);
    return 0;
}
