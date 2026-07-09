#include "ns_gate_registry.hpp"

#include <mutex>
#include <string>
#include <unordered_map>

namespace {

struct RegistryEntry {
    void* owner = nullptr;
    NsGateAckCallback callback = nullptr;
};

std::mutex& registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, RegistryEntry>& registry_entries() {
    static std::unordered_map<std::string, RegistryEntry> entries;
    return entries;
}

}  // namespace

bool ns_gate_registry_register(const char* gate_id, void* owner, NsGateAckCallback callback) {
    if (gate_id == nullptr || gate_id[0] == '\0' || owner == nullptr || callback == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> guard(registry_mutex());
    auto& entries = registry_entries();
    auto found = entries.find(gate_id);
    if (found != entries.end() && found->second.owner != owner) {
        return false;
    }
    entries[gate_id] = RegistryEntry{owner, callback};
    return true;
}

void ns_gate_registry_unregister(const char* gate_id, void* owner) {
    if (gate_id == nullptr || gate_id[0] == '\0' || owner == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> guard(registry_mutex());
    auto& entries = registry_entries();
    auto found = entries.find(gate_id);
    if (found != entries.end() && found->second.owner == owner) {
        entries.erase(found);
    }
}

bool ns_gate_registry_ack(
    const char* gate_id,
    std::uint64_t session_epoch,
    std::uint64_t generation,
    std::uint64_t ack_ns
) {
    RegistryEntry entry;
    {
        std::lock_guard<std::mutex> guard(registry_mutex());
        auto found = registry_entries().find(gate_id == nullptr ? "" : gate_id);
        if (found == registry_entries().end()) {
            return false;
        }
        entry = found->second;
    }
    return entry.callback(entry.owner, session_epoch, generation, ack_ns);
}
