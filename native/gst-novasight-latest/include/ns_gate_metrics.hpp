#pragma once

#include <cstdint>

namespace novasight::latest {

struct NsGateMetrics {
    std::uint64_t received_total = 0;
    std::uint64_t replaced_total = 0;
    std::uint64_t dispatched_total = 0;
    std::uint64_t acked_total = 0;
    std::uint64_t ack_timeout_total = 0;
    std::uint64_t ack_mismatch_total = 0;
    std::uint64_t unexpected_ack_total = 0;
    std::uint64_t eos_total = 0;
    std::uint64_t flush_total = 0;
    std::uint64_t restart_epoch_total = 0;

    std::uint32_t pending_depth = 0;
    std::uint32_t inflight_depth = 0;
    std::uint32_t max_pending_depth = 0;
    std::uint32_t max_inflight_depth = 0;
};

}  // namespace novasight::latest
