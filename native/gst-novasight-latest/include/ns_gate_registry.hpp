#pragma once

#include <cstdint>

using NsGateAckCallback = bool (*)(
    void* owner,
    std::uint64_t session_epoch,
    std::uint64_t generation,
    std::uint64_t ack_ns
);

bool ns_gate_registry_register(const char* gate_id, void* owner, NsGateAckCallback callback);
void ns_gate_registry_unregister(const char* gate_id, void* owner);
bool ns_gate_registry_ack(
    const char* gate_id,
    std::uint64_t session_epoch,
    std::uint64_t generation,
    std::uint64_t ack_ns
);
