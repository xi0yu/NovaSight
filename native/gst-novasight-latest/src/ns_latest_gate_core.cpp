#include "ns_latest_gate_core.hpp"

#include <algorithm>

namespace novasight::latest {

NsLatestGateCore::NsLatestGateCore(NsGateConfig config)
    : config_(config) {
    update_depth_metrics();
}

NsFrameToken NsLatestGateCore::submit(NsGateInput input) {
    if (flushing_ || stopping_ || error_) {
        return NsFrameToken{
            session_epoch_,
            received_generation_,
            input.source_sequence,
            input.source_pts_ns,
            input.gate_arrival_ns,
            0,
            0,
        };
    }

    received_generation_ += 1;
    NsFrameToken token{
        session_epoch_,
        received_generation_,
        input.source_sequence,
        input.source_pts_ns,
        input.gate_arrival_ns,
        0,
        0,
    };

    if (pending_.has_value()) {
        metrics_.replaced_total += 1;
    }
    pending_ = token;
    metrics_.received_total += 1;
    update_depth_metrics();
    return token;
}

std::optional<NsFrameToken> NsLatestGateCore::dispatch_next(std::uint64_t dispatch_ns) {
    if (flushing_ || stopping_ || error_ || !inference_credit_ || inflight_.has_value()) {
        return std::nullopt;
    }
    if (!pending_.has_value()) {
        return std::nullopt;
    }

    NsFrameToken token = *pending_;
    pending_.reset();
    token.gate_dispatch_ns = dispatch_ns;
    inflight_ = token;
    inference_credit_ = false;
    dispatched_generation_ = token.generation;
    inflight_dispatch_ns_ = dispatch_ns;
    metrics_.dispatched_total += 1;
    update_depth_metrics();
    return token;
}

NsAckResult NsLatestGateCore::ack(
    std::uint64_t session_epoch,
    std::uint64_t generation,
    std::uint64_t ack_ns
) {
    if (flushing_) {
        metrics_.unexpected_ack_total += 1;
        return NsAckResult::IgnoredFlushing;
    }
    if (stopping_ || error_) {
        metrics_.unexpected_ack_total += 1;
        return NsAckResult::IgnoredStopping;
    }
    if (!inflight_.has_value()) {
        metrics_.unexpected_ack_total += 1;
        return NsAckResult::Unexpected;
    }
    if (session_epoch != inflight_->session_epoch || generation != inflight_->generation) {
        metrics_.ack_mismatch_total += 1;
        enter_error(NsGateError::AckMismatch);
        return NsAckResult::Mismatch;
    }

    inflight_->infer_ack_ns = ack_ns;
    acked_generation_ = generation;
    inflight_.reset();
    inflight_dispatch_ns_ = 0;
    inference_credit_ = true;
    metrics_.acked_total += 1;
    update_depth_metrics();
    return NsAckResult::Accepted;
}

bool NsLatestGateCore::check_ack_timeout(std::uint64_t now_ns) {
    if (!inflight_.has_value() || error_ || flushing_ || stopping_) {
        return false;
    }
    if (now_ns <= inflight_dispatch_ns_) {
        return false;
    }
    if ((now_ns - inflight_dispatch_ns_) <= config_.ack_timeout_ns) {
        return false;
    }
    metrics_.ack_timeout_total += 1;
    enter_error(NsGateError::AckTimeout);
    return true;
}

void NsLatestGateCore::mark_push_failed() {
    enter_error(NsGateError::PushFailed);
}

void NsLatestGateCore::flush_start() {
    flushing_ = true;
    pending_.reset();
    metrics_.flush_total += 1;
    update_depth_metrics();
}

void NsLatestGateCore::flush_stop() {
    pending_.reset();
    inflight_.reset();
    flushing_ = false;
    stopping_ = false;
    error_ = false;
    error_code_ = NsGateError::None;
    inference_credit_ = true;
    inflight_dispatch_ns_ = 0;
    session_epoch_ += 1;
    received_generation_ = 0;
    dispatched_generation_ = 0;
    acked_generation_ = 0;
    metrics_.restart_epoch_total += 1;
    update_depth_metrics();
}

std::optional<NsFrameToken> NsLatestGateCore::eos() {
    metrics_.eos_total += 1;
    if (config_.eos_policy == NsEosPolicy::DropPending) {
        pending_.reset();
        update_depth_metrics();
        return std::nullopt;
    }
    if (inflight_.has_value() || !inference_credit_ || !pending_.has_value()) {
        update_depth_metrics();
        return std::nullopt;
    }
    return dispatch_next(pending_->gate_arrival_ns);
}

void NsLatestGateCore::stop() {
    stopping_ = true;
    pending_.reset();
    update_depth_metrics();
}

void NsLatestGateCore::reset() {
    pending_.reset();
    inflight_.reset();
    metrics_ = NsGateMetrics{};
    inference_credit_ = true;
    flushing_ = false;
    stopping_ = false;
    error_ = false;
    error_code_ = NsGateError::None;
    session_epoch_ = 1;
    received_generation_ = 0;
    dispatched_generation_ = 0;
    acked_generation_ = 0;
    inflight_dispatch_ns_ = 0;
    update_depth_metrics();
}

NsGateSnapshot NsLatestGateCore::snapshot() const {
    return NsGateSnapshot{
        pending_.has_value(),
        inference_credit_,
        inflight_.has_value(),
        flushing_,
        stopping_,
        error_,
        error_code_,
        session_epoch_,
        received_generation_,
        dispatched_generation_,
        acked_generation_,
        inflight_dispatch_ns_,
        metrics_,
    };
}

std::string NsLatestGateCore::error_string() const {
    switch (error_code_) {
        case NsGateError::None:
            return "";
        case NsGateError::AckMismatch:
            return "ACK_MISMATCH";
        case NsGateError::AckTimeout:
            return "ACK_TIMEOUT";
        case NsGateError::PushFailed:
            return "PUSH_FAILED";
    }
    return "UNKNOWN";
}

void NsLatestGateCore::update_depth_metrics() {
    metrics_.pending_depth = pending_.has_value() ? 1U : 0U;
    metrics_.inflight_depth = inflight_.has_value() ? 1U : 0U;
    metrics_.max_pending_depth = std::max(metrics_.max_pending_depth, metrics_.pending_depth);
    metrics_.max_inflight_depth = std::max(metrics_.max_inflight_depth, metrics_.inflight_depth);
}

void NsLatestGateCore::enter_error(NsGateError error) {
    error_ = true;
    error_code_ = error;
    inference_credit_ = false;
    pending_.reset();
    update_depth_metrics();
}

}  // namespace novasight::latest
