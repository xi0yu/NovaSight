#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "ns_frame_token.hpp"
#include "ns_gate_metrics.hpp"

namespace novasight::latest {

enum class NsGateError {
    None,
    AckMismatch,
    AckTimeout,
    PushFailed,
};

enum class NsAckResult {
    Accepted,
    Unexpected,
    Mismatch,
    IgnoredFlushing,
    IgnoredStopping,
};

enum class NsEosPolicy {
    DrainLatest,
    DropPending,
};

struct NsGateConfig {
    std::uint64_t ack_timeout_ns = 200'000'000ULL;
    NsEosPolicy eos_policy = NsEosPolicy::DrainLatest;
};

struct NsGateInput {
    std::uint64_t source_sequence = 0;
    std::uint64_t source_pts_ns = 0;
    std::uint64_t gate_arrival_ns = 0;
};

struct NsGateSnapshot {
    bool has_pending = false;
    bool inference_credit = true;
    bool inference_inflight = false;
    bool flushing = false;
    bool stopping = false;
    bool error = false;
    NsGateError error_code = NsGateError::None;

    std::uint64_t session_epoch = 0;
    std::uint64_t received_generation = 0;
    std::uint64_t dispatched_generation = 0;
    std::uint64_t acked_generation = 0;
    std::uint64_t inflight_dispatch_ns = 0;
    NsGateMetrics metrics;
};

class NsLatestGateCore {
public:
    explicit NsLatestGateCore(NsGateConfig config = {});

    NsFrameToken submit(NsGateInput input);
    std::optional<NsFrameToken> dispatch_next(std::uint64_t dispatch_ns);
    NsAckResult ack(
        std::uint64_t session_epoch,
        std::uint64_t generation,
        std::uint64_t ack_ns
    );
    bool check_ack_timeout(std::uint64_t now_ns);

    void mark_push_failed();
    void flush_start();
    void flush_stop();
    std::optional<NsFrameToken> eos();
    void stop();
    void reset();

    NsGateSnapshot snapshot() const;
    std::string error_string() const;

private:
    void update_depth_metrics();
    void enter_error(NsGateError error);

    NsGateConfig config_;
    std::optional<NsFrameToken> pending_;
    std::optional<NsFrameToken> inflight_;
    NsGateMetrics metrics_;

    bool inference_credit_ = true;
    bool flushing_ = false;
    bool stopping_ = false;
    bool error_ = false;
    NsGateError error_code_ = NsGateError::None;

    std::uint64_t session_epoch_ = 1;
    std::uint64_t received_generation_ = 0;
    std::uint64_t dispatched_generation_ = 0;
    std::uint64_t acked_generation_ = 0;
    std::uint64_t inflight_dispatch_ns_ = 0;
};

}  // namespace novasight::latest
