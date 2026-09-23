#include "latest_frame_exchange.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

using novasight::exchange::FrameContent;
using novasight::exchange::FrameDescriptor;
using novasight::exchange::FrameMemory;
using novasight::exchange::LatestFrameExchange;

namespace {

FrameDescriptor frame(std::uint64_t generation, std::vector<int>* released) {
    return FrameDescriptor{
        generation,
        generation,
        generation,
        10'000 + generation,
        20'000 + generation,
        960,
        960,
        960,
        "NV12",
        "monotonic",
        FrameContent::VideoSurface,
        {reinterpret_cast<void*>(generation), "GstBuffer", FrameMemory::Nvmm, [released](void* ptr) {
             released->push_back(static_cast<int>(reinterpret_cast<std::uintptr_t>(ptr)));
         }},
    };
}

}  // namespace

int main() {
    LatestFrameExchange exchange;
    std::vector<int> released;

    exchange.publish(frame(1, &released));
    auto current = exchange.acquire_latest_after(0);
    assert(current);
    assert(current->descriptor().generation == 1);
    assert(released.empty());

    exchange.publish(frame(2, &released));
    exchange.publish(frame(3, &released));

    auto status = exchange.status();
    assert(status.pending_depth == 1);
    assert(status.max_pending_depth == 1);
    assert(status.published_total == 3);
    assert(status.acquired_total == 1);
    assert(status.replaced_total == 1);
    assert((released == std::vector<int>{2}));

    auto next = exchange.acquire_latest_after(current->descriptor().generation);
    assert(next);
    assert(next->descriptor().generation == 3);
    assert(current->descriptor().generation == 1);

    current.reset();
    assert((released == std::vector<int>{2, 1}));

    next.reset();
    assert((released == std::vector<int>{2, 1, 3}));
    return 0;
}
