#pragma once

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

namespace novasight {

// TensorRT may log from internal threads. Keep the first SDK error for the
// current operation; later cleanup errors must not replace its cause.
class TensorRtErrorLog {
public:
    void record(const char* message) noexcept {
        if (!message || !*message) return;
        try {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (*message_) return;
            const int length = std::snprintf(message_, sizeof(message_), "%s", message);
            if (length >= static_cast<int>(sizeof(message_))) {
                constexpr char tail[] = "... [truncated]";
                std::memcpy(message_ + sizeof(message_) - sizeof(tail), tail, sizeof(tail));
            }
        } catch (...) {
            // ILogger::log is noexcept; even a locking failure must preserve
            // diagnostics without throwing through the SDK callback.
            std::fprintf(stderr, "TensorRT diagnostic capture failed: %s\n", message);
        }
    }

    void clear() {
        const std::lock_guard<std::mutex> lock(mutex_);
        message_[0] = '\0';
    }

    std::string describe(const char* operation) const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return *message_ ? std::string(operation) + "; TensorRT: " + message_ : operation;
    }

private:
    mutable std::mutex mutex_;
    // Bounded to leave room for operation context in the 1024-byte FFI error.
    char message_[768]{};
};

}  // namespace novasight
