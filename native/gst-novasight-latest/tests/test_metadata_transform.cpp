#include "ns_latest_gate_core.hpp"

#include <cassert>

using novasight::latest::NsAckResult;
using novasight::latest::NsGateError;
using novasight::latest::NsGateInput;
using novasight::latest::NsLatestGateCore;

int main() {
    NsLatestGateCore gate;

    auto token = gate.submit(NsGateInput{42, 12'000, 34'000});
    assert(token.session_epoch == 1);
    assert(token.generation == 1);
    assert(token.source_sequence == 42);
    assert(token.source_pts_ns == 12'000);
    assert(token.gate_arrival_ns == 34'000);

    auto dispatched = gate.dispatch_next(40'000);
    assert(dispatched.has_value());

    auto wrong_generation = dispatched->generation + 1;
    assert(gate.ack(dispatched->session_epoch, wrong_generation, 45'000) == NsAckResult::Mismatch);
    auto snapshot = gate.snapshot();
    assert(snapshot.error);
    assert(snapshot.error_code == NsGateError::AckMismatch);
    assert(snapshot.metrics.ack_mismatch_total == 1);
    assert(!snapshot.inference_credit);
    return 0;
}
