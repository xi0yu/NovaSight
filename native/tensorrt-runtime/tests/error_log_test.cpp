#include "novasight_tensorrt_error.hpp"

#include <cassert>
#include <string>
#include <thread>
#include <vector>

int main() {
    novasight::TensorRtErrorLog log;
    log.record(nullptr);
    log.record("");
    assert(log.describe("deserialize") == "deserialize");
    log.record("engine version mismatch");
    std::vector<std::thread> writers;
    for (int i = 0; i < 8; ++i)
        writers.emplace_back([&] { log.record("secondary failure"); });
    for (auto& writer : writers) writer.join();
    assert(log.describe("deserializeCudaEngine returned null") ==
        "deserializeCudaEngine returned null; TensorRT: engine version mismatch");
    log.clear();
    assert(log.describe("enqueue") == "enqueue");
    log.record(std::string(4096, 'x').c_str());
    const auto detail = log.describe("enqueue");
    assert(detail.size() < 1024);
    assert(detail.find("[truncated]") != std::string::npos);
}
