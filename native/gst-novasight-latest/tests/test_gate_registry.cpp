#include "ns_gate_registry.hpp"

#include <cassert>
#include <cstdint>

namespace {

struct AckState {
    std::uint64_t epoch = 0;
    std::uint64_t generation = 0;
    std::uint64_t ack_ns = 0;
    int calls = 0;
};

bool ack_callback(
    void* owner,
    std::uint64_t session_epoch,
    std::uint64_t generation,
    std::uint64_t ack_ns
) {
    auto* state = static_cast<AckState*>(owner);
    state->epoch = session_epoch;
    state->generation = generation;
    state->ack_ns = ack_ns;
    state->calls += 1;
    return true;
}

bool reject_callback(
    void* owner,
    std::uint64_t session_epoch,
    std::uint64_t generation,
    std::uint64_t ack_ns
) {
    (void)owner;
    (void)session_epoch;
    (void)generation;
    (void)ack_ns;
    return false;
}

}  // namespace

int main() {
    AckState first;
    AckState second;

    assert(ns_gate_registry_register("gate-a", &first, ack_callback));
    assert(!ns_gate_registry_register("gate-a", &second, ack_callback));
    assert(ns_gate_registry_ack("gate-a", 7, 11, 13));
    assert(first.calls == 1);
    assert(first.epoch == 7);
    assert(first.generation == 11);
    assert(first.ack_ns == 13);

    ns_gate_registry_unregister("gate-a", &second);
    assert(ns_gate_registry_ack("gate-a", 8, 12, 14));
    assert(first.calls == 2);

    ns_gate_registry_unregister("gate-a", &first);
    assert(!ns_gate_registry_ack("gate-a", 1, 1, 1));

    assert(ns_gate_registry_register("gate-b", &second, reject_callback));
    assert(!ns_gate_registry_ack("gate-b", 1, 2, 3));
    ns_gate_registry_unregister("gate-b", &second);
    return 0;
}
