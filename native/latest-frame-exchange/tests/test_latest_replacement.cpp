#include "latest_frame_exchange.hpp"

#include <cassert>
#include <vector>

using novasight::exchange::FrameDescriptor;
using novasight::exchange::FrameContent;
using novasight::exchange::FrameMemory;
using novasight::exchange::LatestFrameExchange;

namespace {

FrameDescriptor frame(std::uint64_t generation, std::uint64_t sequence, std::vector<int>* released) {
    return FrameDescriptor{
        generation,
        sequence,
        sequence,
        1'000 + generation,
        2'000 + generation,
        640,
        640,
        640,
        "NV12",
        "monotonic",
        FrameContent::VideoSurface,
        {reinterpret_cast<void*>(sequence), "GstBuffer", FrameMemory::Nvmm, [released](void* ptr) {
             released->push_back(static_cast<int>(reinterpret_cast<std::uintptr_t>(ptr)));
         }},
    };
}

}  // namespace

int main() {
    LatestFrameExchange exchange;
    std::vector<int> released;

    exchange.publish(frame(1, 100, &released));
    exchange.publish(frame(2, 101, &released));
    exchange.publish(frame(3, 102, &released));

    auto selected = exchange.acquire_latest_after(0);
    assert(selected);
    assert(selected->descriptor().generation == 3);
    assert(selected->descriptor().source_sequence == 102);
    assert(released.size() == 2);
    assert(released[0] == 100);
    assert(released[1] == 101);

    auto status = exchange.status();
    assert(status.pending_depth == 0);
    assert(status.max_pending_depth == 1);
    assert(status.published_total == 3);
    assert(status.acquired_total == 1);
    assert(status.replaced_total == 2);
    return 0;
}
