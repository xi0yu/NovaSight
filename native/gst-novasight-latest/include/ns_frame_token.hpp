#pragma once

#include <cstdint>

namespace novasight::latest {

struct NsFrameToken {
    std::uint64_t session_epoch = 0;
    std::uint64_t generation = 0;
    std::uint64_t source_sequence = 0;
    std::uint64_t source_pts_ns = 0;
    std::uint64_t gate_arrival_ns = 0;
    std::uint64_t gate_dispatch_ns = 0;
    std::uint64_t infer_ack_ns = 0;
};

}  // namespace novasight::latest
