#include "ns_latest_gate_core.hpp"

#include <cassert>

using novasight::latest::NsAckResult;
using novasight::latest::NsGateConfig;
using novasight::latest::NsGateError;
using novasight::latest::NsGateInput;
using novasight::latest::NsLatestGateCore;

int main() {
    NsLatestGateCore gate(NsGateConfig{100});

    gate.submit(NsGateInput{1, 0, 1'000});
    auto first = gate.dispatch_next(2'000);
    assert(first.has_value());

    assert(!gate.check_ack_timeout(2'050));
    assert(gate.check_ack_timeout(2'101));

    auto snapshot = gate.snapshot();
    assert(snapshot.error);
    assert(snapshot.error_code == NsGateError::AckTimeout);
    assert(snapshot.metrics.ack_timeout_total == 1);
    assert(snapshot.metrics.pending_depth == 0);
    assert(snapshot.metrics.inflight_depth == 1);
    assert(!snapshot.inference_credit);

    gate.submit(NsGateInput{2, 0, 3'000});
    auto blocked = gate.dispatch_next(3'100);
    assert(!blocked.has_value());
    assert(gate.ack(first->session_epoch, first->generation, 3'200) == NsAckResult::IgnoredStopping);
    return 0;
}
