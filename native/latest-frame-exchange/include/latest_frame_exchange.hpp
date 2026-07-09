#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace novasight::exchange {

enum class FrameMemory {
    Unknown,
    Cpu,
    Nvmm,
    CudaDevice,
};

struct FrameResource {
    void* handle = nullptr;
    std::string kind;
    FrameMemory memory = FrameMemory::Unknown;
    std::function<void(void*)> release;
};

struct FrameDescriptor {
    std::uint64_t generation = 0;
    std::uint64_t frame_id = 0;
    std::uint64_t source_sequence = 0;
    std::uint64_t capture_ts_ns = 0;
    std::uint64_t pipeline_running_time_ns = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t pitch = 0;
    std::string format;
    std::string clock_domain = "monotonic";
    FrameResource resource;
};

class FrameHandle {
public:
    explicit FrameHandle(FrameDescriptor descriptor);
    FrameHandle(const FrameHandle&) = delete;
    FrameHandle& operator=(const FrameHandle&) = delete;
    FrameHandle(FrameHandle&&) noexcept = default;
    FrameHandle& operator=(FrameHandle&&) noexcept = default;
    ~FrameHandle();

    const FrameDescriptor& descriptor() const;
    void release();

private:
    FrameDescriptor descriptor_;
    bool released_ = false;
};

struct ExchangeStatus {
    std::uint32_t pending_depth = 0;
    std::uint32_t max_pending_depth = 1;
    std::uint64_t published_generation = 0;
    std::uint64_t acquired_generation = 0;
    std::uint64_t published_total = 0;
    std::uint64_t acquired_total = 0;
    std::uint64_t replaced_total = 0;
};

class LatestFrameExchange {
public:
    void publish(FrameDescriptor descriptor);
    std::shared_ptr<FrameHandle> acquire_latest_after(std::uint64_t after_generation);
    void clear();
    ExchangeStatus status() const;

private:
    mutable std::mutex mutex_;
    std::shared_ptr<FrameHandle> pending_;
    ExchangeStatus status_;
};

}  // namespace novasight::exchange
