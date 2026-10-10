#include "latest_frame_exchange.hpp"

#include <cassert>
#include <stdexcept>

using novasight::exchange::FrameContent;
using novasight::exchange::FrameDescriptor;
using novasight::exchange::FrameMemory;
using novasight::exchange::LatestFrameExchange;

int main() {
    LatestFrameExchange exchange;
    FrameDescriptor tensor_like{
        1,
        1,
        1,
        1'000,
        0,
        320,
        320,
        0,
        "NCHW_FP16",
        "monotonic",
        FrameContent::Tensor,
        {nullptr, "TensorRTInputBuffer", FrameMemory::CudaDevice, {}},
    };

    bool rejected = false;
    try {
        exchange.publish(std::move(tensor_like));
    } catch (const std::invalid_argument& exc) {
        rejected = std::string(exc.what()).find("video surfaces") != std::string::npos;
    }
    assert(rejected);
    return 0;
}
