#include "ns_latest_gate_core.hpp"

#include <cassert>

using novasight::latest::NsAckResult;
using novasight::latest::NsGateInput;
using novasight::latest::NsLatestGateCore;

int main() {
    NsLatestGateCore gate;

    gate.submit(NsGateInput{100, 1'000, 10'000});
    auto first = gate.dispatch_next(11'000);
    assert(first.has_value());
    assert(first->generation == 1);
    assert(first->source_sequence == 100);

    for (std::uint64_t seq = 101; seq <= 108; ++seq) {
        gate.submit(NsGateInput{seq, seq * 10, seq * 100});
    }

    auto blocked = gate.dispatch_next(20'000);
    assert(!blocked.has_value());

    auto snapshot = gate.snapshot();
    assert(snapshot.metrics.pending_depth == 1);
    assert(snapshot.metrics.inflight_depth == 1);
    assert(snapshot.metrics.max_pending_depth == 1);
    assert(snapshot.metrics.max_inflight_depth == 1);
    assert(snapshot.metrics.replaced_total == 7);

    assert(gate.ack(first->session_epoch, first->generation, 30'000) == NsAckResult::Accepted);
    auto latest = gate.dispatch_next(31'000);
    assert(latest.has_value());
    assert(latest->source_sequence == 108);
    assert(latest->generation == 9);

    snapshot = gate.snapshot();
    assert(snapshot.metrics.dispatched_total == 2);
    assert(snapshot.metrics.acked_total == 1);
    assert(snapshot.metrics.pending_depth == 0);
    assert(snapshot.metrics.inflight_depth == 1);
    return 0;
}
