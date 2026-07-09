#include "latest_frame_exchange.hpp"

#include <cassert>
#include <vector>

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
        640,
        640,
        640,
        "NV12",
        "monotonic",
        {nullptr, "GstBuffer", FrameMemory::Nvmm, {}},
    };
}

}  // namespace

int main() {
    LatestFrameExchange exchange;
    std::vector<std::uint64_t> inferred;
    std::uint64_t last_generation = 0;

    exchange.publish(frame(100));
    auto first = exchange.acquire_latest_after(last_generation);
    assert(first);
    inferred.push_back(first->descriptor().generation);
    last_generation = first->descriptor().generation;

    for (std::uint64_t generation = 101; generation <= 108; ++generation) {
        exchange.publish(frame(generation));
    }

    auto second = exchange.acquire_latest_after(last_generation);
    assert(second);
    inferred.push_back(second->descriptor().generation);
    last_generation = second->descriptor().generation;

    for (std::uint64_t generation = 109; generation <= 116; ++generation) {
        exchange.publish(frame(generation));
    }

    auto third = exchange.acquire_latest_after(last_generation);
    assert(third);
    inferred.push_back(third->descriptor().generation);

    assert((inferred == std::vector<std::uint64_t>{100, 108, 116}));
    auto status = exchange.status();
    assert(status.replaced_total == 14);
    assert(status.acquired_total == 3);
    return 0;
}
