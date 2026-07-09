#include "ns_latest_gate_core.hpp"

#include <cassert>

using novasight::latest::NsEosPolicy;
using novasight::latest::NsGateConfig;
using novasight::latest::NsGateInput;
using novasight::latest::NsLatestGateCore;

int main() {
    NsLatestGateCore drain_gate(NsGateConfig{200'000'000, NsEosPolicy::DrainLatest});
    drain_gate.submit(NsGateInput{7, 0, 1'000});
    auto drained = drain_gate.eos();
    assert(drained.has_value());
    assert(drained->source_sequence == 7);
    auto drain_snapshot = drain_gate.snapshot();
    assert(drain_snapshot.metrics.eos_total == 1);
    assert(drain_snapshot.metrics.pending_depth == 0);
    assert(drain_snapshot.metrics.inflight_depth == 1);

    NsLatestGateCore drop_gate(NsGateConfig{200'000'000, NsEosPolicy::DropPending});
    drop_gate.submit(NsGateInput{8, 0, 2'000});
    auto dropped = drop_gate.eos();
    assert(!dropped.has_value());
    auto drop_snapshot = drop_gate.snapshot();
    assert(drop_snapshot.metrics.eos_total == 1);
    assert(drop_snapshot.metrics.pending_depth == 0);
    assert(drop_snapshot.metrics.inflight_depth == 0);
    return 0;
}
