#include "ns_latest_gate_core.hpp"

#include <cassert>

using novasight::latest::NsAckResult;
using novasight::latest::NsGateInput;
using novasight::latest::NsLatestGateCore;

int main() {
    NsLatestGateCore gate;

    gate.submit(NsGateInput{1, 0, 1'000});
    auto first = gate.dispatch_next(2'000);
    assert(first.has_value());
    const auto old_epoch = first->session_epoch;

    gate.submit(NsGateInput{2, 0, 3'000});
    gate.flush_start();
    auto snapshot = gate.snapshot();
    assert(snapshot.flushing);
    assert(snapshot.metrics.pending_depth == 0);
    assert(snapshot.metrics.inflight_depth == 1);
    assert(gate.ack(old_epoch, first->generation, 4'000) == NsAckResult::IgnoredFlushing);

    gate.flush_stop();
    snapshot = gate.snapshot();
    assert(!snapshot.flushing);
    assert(snapshot.session_epoch == old_epoch + 1);
    assert(snapshot.inference_credit);
    assert(!snapshot.inference_inflight);
    assert(snapshot.metrics.restart_epoch_total == 1);

    gate.submit(NsGateInput{10, 0, 5'000});
    auto next = gate.dispatch_next(6'000);
    assert(next.has_value());
    assert(next->session_epoch == old_epoch + 1);
    assert(next->generation == 1);
    assert(gate.ack(old_epoch, first->generation, 7'000) == NsAckResult::Mismatch);
    assert(gate.snapshot().metrics.ack_mismatch_total == 1);
    return 0;
}
