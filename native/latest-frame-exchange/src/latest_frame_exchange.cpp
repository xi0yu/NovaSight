#include "latest_frame_exchange.hpp"

#include <stdexcept>

namespace novasight::exchange {

namespace {

void validate_descriptor(const FrameDescriptor& descriptor) {
    if (descriptor.generation == 0) {
        throw std::invalid_argument("FrameDescriptor.generation must be positive");
    }
    if (descriptor.capture_ts_ns == 0) {
        throw std::invalid_argument("FrameDescriptor.capture_ts_ns must be positive");
    }
    if (descriptor.width == 0 || descriptor.height == 0) {
        throw std::invalid_argument("FrameDescriptor width/height must be positive");
    }
    if (descriptor.format.empty()) {
        throw std::invalid_argument("FrameDescriptor.format must be non-empty");
    }
    if (descriptor.clock_domain.empty()) {
        throw std::invalid_argument("FrameDescriptor.clock_domain must be non-empty");
    }
}

}  // namespace

FrameHandle::FrameHandle(FrameDescriptor descriptor)
    : descriptor_(std::move(descriptor)) {
    validate_descriptor(descriptor_);
}

FrameHandle::~FrameHandle() {
    release();
}

const FrameDescriptor& FrameHandle::descriptor() const {
    return descriptor_;
}

void FrameHandle::release() {
    if (released_) {
        return;
    }
    released_ = true;
    if (descriptor_.resource.release) {
        descriptor_.resource.release(descriptor_.resource.handle);
    }
}

void LatestFrameExchange::publish(FrameDescriptor descriptor) {
    auto next = std::make_shared<FrameHandle>(std::move(descriptor));
    std::shared_ptr<FrameHandle> old;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (pending_) {
            old = std::move(pending_);
            status_.replaced_total += 1;
        }
        pending_ = std::move(next);
        status_.pending_depth = 1;
        status_.published_total += 1;
        status_.published_generation = pending_->descriptor().generation;
    }
    if (old) {
        old->release();
    }
}

std::shared_ptr<FrameHandle> LatestFrameExchange::acquire_latest_after(
    std::uint64_t after_generation
) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!pending_) {
        return nullptr;
    }
    if (pending_->descriptor().generation <= after_generation) {
        return nullptr;
    }
    auto selected = std::move(pending_);
    status_.pending_depth = 0;
    status_.acquired_total += 1;
    status_.acquired_generation = selected->descriptor().generation;
    return selected;
}

void LatestFrameExchange::clear() {
    std::shared_ptr<FrameHandle> old;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        old = std::move(pending_);
        status_.pending_depth = 0;
    }
    if (old) {
        old->release();
    }
}

ExchangeStatus LatestFrameExchange::status() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return status_;
}

}  // namespace novasight::exchange
