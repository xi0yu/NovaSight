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

    gate.submit(NsGateInput{2, 0, 3'000});
    auto second_without_ack = gate.dispatch_next(4'000);
    assert(!second_without_ack.has_value());

    auto snapshot = gate.snapshot();
    assert(!snapshot.inference_credit);
    assert(snapshot.inference_inflight);
    assert(snapshot.metrics.pending_depth == 1);
    assert(snapshot.metrics.inflight_depth == 1);

    assert(gate.ack(first->session_epoch, first->generation, 5'000) == NsAckResult::Accepted);
    auto second = gate.dispatch_next(6'000);
    assert(second.has_value());
    assert(second->source_sequence == 2);
    assert(second->generation == 2);

    snapshot = gate.snapshot();
    assert(snapshot.metrics.max_pending_depth == 1);
    assert(snapshot.metrics.max_inflight_depth == 1);
    return 0;
}
