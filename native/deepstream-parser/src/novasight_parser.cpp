#include <nvdsinfer_custom_impl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kInitialObjectCapacity = 256U;
constexpr std::size_t kRockchipAnchorCount = 3U;
constexpr float kRockchipYoloV5Anchors[3][kRockchipAnchorCount][2] = {
    {{10.0F, 13.0F}, {16.0F, 30.0F}, {33.0F, 23.0F}},
    {{30.0F, 61.0F}, {62.0F, 45.0F}, {59.0F, 119.0F}},
    {{116.0F, 90.0F}, {156.0F, 198.0F}, {373.0F, 326.0F}},
};

std::atomic<std::uint64_t> g_decode_calls{0};
std::atomic<std::uint64_t> g_parse_failures{0};
std::atomic<std::uint64_t> g_last_error_code{0};
std::atomic<std::uint64_t> g_last_decode_ns{0};
std::atomic<std::uint64_t> g_last_input_candidates{0};
std::atomic<std::uint64_t> g_last_output_candidates{0};

float half_to_float(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    std::int32_t exponent = static_cast<std::int32_t>((value >> 10U) & 0x1FU);
    std::uint32_t mantissa = value & 0x03FFU;
    std::uint32_t bits = 0;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            exponent = 1U;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= 0x03FFU;
            bits = sign
                | (static_cast<std::uint32_t>(exponent + 112) << 23U)
                | (mantissa << 13U);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        bits = sign
            | (static_cast<std::uint32_t>(exponent + 112) << 23U)
            | (mantissa << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float layer_value(const NvDsInferLayerInfo& layer, std::size_t index) {
    if (layer.dataType == FLOAT) {
        return static_cast<const float*>(layer.buffer)[index];
    }
    if (layer.dataType == HALF) {
        return half_to_float(static_cast<const std::uint16_t*>(layer.buffer)[index]);
    }
    if (layer.dataType == INT32) {
        return static_cast<float>(static_cast<const std::int32_t*>(layer.buffer)[index]);
    }
    return std::numeric_limits<float>::quiet_NaN();
}

float threshold_for_class(
    const NvDsInferParseDetectionParams& params,
    std::size_t class_id);

std::size_t layer_element_count(const NvDsInferLayerInfo& layer) {
    std::size_t count = 1U;
    if (layer.inferDims.numDims == 0U) {
        return 0U;
    }
    for (unsigned int index = 0; index < layer.inferDims.numDims; ++index) {
        if (layer.inferDims.d[index] <= 0) {
            return 0U;
        }
        count *= static_cast<std::size_t>(layer.inferDims.d[index]);
    }
    return count;
}

bool layer_name_contains(const NvDsInferLayerInfo& layer, const char* token) {
    if (layer.layerName == nullptr) {
        return false;
    }
    std::string name(layer.layerName);
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return name.find(token) != std::string::npos;
}

bool parse_efficient_nms(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects,
    std::size_t& input_candidates) {
    const NvDsInferLayerInfo* count_layer = nullptr;
    const NvDsInferLayerInfo* boxes_layer = nullptr;
    const NvDsInferLayerInfo* scores_layer = nullptr;
    const NvDsInferLayerInfo* classes_layer = nullptr;
    for (const auto& layer : layers) {
        if (layer.buffer == nullptr) {
            return false;
        }
        if (layer_name_contains(layer, "num_det")
            || layer_name_contains(layer, "count")) {
            count_layer = &layer;
        } else if (layer_name_contains(layer, "box")
            || layer_name_contains(layer, "bbox")) {
            boxes_layer = &layer;
        } else if (layer_name_contains(layer, "score")
            || layer_name_contains(layer, "conf")) {
            scores_layer = &layer;
        } else if (layer_name_contains(layer, "class")
            || layer_name_contains(layer, "label")) {
            classes_layer = &layer;
        }
    }
    if (count_layer == nullptr || boxes_layer == nullptr || scores_layer == nullptr
        || classes_layer == nullptr || layer_element_count(*count_layer) != 1U) {
        return false;
    }
    const std::size_t box_values = layer_element_count(*boxes_layer);
    const std::size_t score_values = layer_element_count(*scores_layer);
    const std::size_t class_values = layer_element_count(*classes_layer);
    if (box_values == 0U || box_values % 4U != 0U
        || score_values != box_values / 4U || class_values != score_values) {
        return false;
    }
    const float raw_count = layer_value(*count_layer, 0U);
    if (!std::isfinite(raw_count) || raw_count < 0.0F) {
        return false;
    }
    input_candidates = std::min(
        static_cast<std::size_t>(raw_count),
        score_values);
    objects.reserve(objects.size() + std::min(input_candidates, kInitialObjectCapacity));
    for (std::size_t index = 0; index < input_candidates; ++index) {
        const float score = layer_value(*scores_layer, index);
        const float raw_class = layer_value(*classes_layer, index);
        if (!std::isfinite(score) || !std::isfinite(raw_class) || raw_class < 0.0F) {
            continue;
        }
        const std::size_t class_id = static_cast<std::size_t>(raw_class);
        if (class_id >= params.numClassesConfigured
            || score < threshold_for_class(params, class_id)) {
            continue;
        }
        float left = layer_value(*boxes_layer, index * 4U);
        float top = layer_value(*boxes_layer, index * 4U + 1U);
        float right = layer_value(*boxes_layer, index * 4U + 2U);
        float bottom = layer_value(*boxes_layer, index * 4U + 3U);
        if (!std::isfinite(left) || !std::isfinite(top) || !std::isfinite(right)
            || !std::isfinite(bottom)) {
            continue;
        }
        const float largest_coordinate = std::max(
            std::max(std::fabs(left), std::fabs(right)),
            std::max(std::fabs(top), std::fabs(bottom)));
        if (largest_coordinate <= 2.0F) {
            left *= static_cast<float>(network.width);
            right *= static_cast<float>(network.width);
            top *= static_cast<float>(network.height);
            bottom *= static_cast<float>(network.height);
        }
        left = std::max(0.0F, left);
        top = std::max(0.0F, top);
        right = std::min(static_cast<float>(network.width), right);
        bottom = std::min(static_cast<float>(network.height), bottom);
        if (right <= left || bottom <= top) {
            continue;
        }
        NvDsInferObjectDetectionInfo object{};
        object.classId = static_cast<unsigned int>(class_id);
        object.detectionConfidence = score;
        object.left = left;
        object.top = top;
        object.width = right - left;
        object.height = bottom - top;
        objects.push_back(object);
    }
    return true;
}

float threshold_for_class(
    const NvDsInferParseDetectionParams& params,
    std::size_t class_id) {
    if (class_id < params.perClassPreclusterThreshold.size()) {
        return params.perClassPreclusterThreshold[class_id];
    }
    return 0.0F;
}

bool output_shape(
    const NvDsInferLayerInfo& layer,
    std::size_t class_count,
    std::size_t& channels,
    std::size_t& candidates,
    bool& channels_first) {
    std::vector<std::size_t> dimensions;
    dimensions.reserve(layer.inferDims.numDims);
    for (unsigned int index = 0; index < layer.inferDims.numDims; ++index) {
        const int value = layer.inferDims.d[index];
        if (value <= 0) {
            return false;
        }
        dimensions.push_back(static_cast<std::size_t>(value));
    }
    if (dimensions.size() == 3U && dimensions.front() == 1U) {
        dimensions.erase(dimensions.begin());
    }
    if (dimensions.size() != 2U) {
        return false;
    }
    const std::size_t without_objectness = class_count + 4U;
    const std::size_t with_objectness = class_count + 5U;
    if (dimensions[0] == without_objectness || dimensions[0] == with_objectness) {
        channels = dimensions[0];
        candidates = dimensions[1];
        channels_first = true;
        return true;
    }
    if (dimensions[1] == without_objectness || dimensions[1] == with_objectness) {
        channels = dimensions[1];
        candidates = dimensions[0];
        channels_first = false;
        return true;
    }
    return false;
}

float prediction_value(
    const NvDsInferLayerInfo& layer,
    std::size_t candidate,
    std::size_t channel,
    std::size_t channels,
    std::size_t candidates,
    bool channels_first) {
    const std::size_t index = channels_first
        ? channel * candidates + candidate
        : candidate * channels + channel;
    return layer_value(layer, index);
}

struct RockchipHead {
    const NvDsInferLayerInfo* layer = nullptr;
    std::size_t channels = 0U;
    std::size_t height = 0U;
    std::size_t width = 0U;
};

// In-graph Decode/NMS engines commonly expose [1,N,6] as
// x1,y1,x2,y2,confidence,class_id. Consume that contract explicitly.
bool parse_decoded_boxes6(
    const NvDsInferLayerInfo& layer,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects,
    std::size_t& input_candidates) {
    const std::size_t count = layer_element_count(layer);
    if (count == 0U || count % 6U != 0U || network.width == 0U || network.height == 0U) return false;
    input_candidates = count / 6U;
    if (input_candidates > 200000U) return false;
    objects.reserve(objects.size() + std::min(input_candidates, kInitialObjectCapacity));
    for (std::size_t i = 0; i < input_candidates; ++i) {
        float left = layer_value(layer, i * 6U), top = layer_value(layer, i * 6U + 1U);
        float right = layer_value(layer, i * 6U + 2U), bottom = layer_value(layer, i * 6U + 3U);
        const float score = layer_value(layer, i * 6U + 4U);
        const float raw_class = layer_value(layer, i * 6U + 5U);
        if (!std::isfinite(left) || !std::isfinite(top) || !std::isfinite(right)
            || !std::isfinite(bottom) || !std::isfinite(score) || !std::isfinite(raw_class)
            || raw_class < 0.0F) continue;
        const std::size_t class_id = static_cast<std::size_t>(raw_class);
        if (class_id >= params.numClassesConfigured || score < threshold_for_class(params, class_id)) continue;
        const float largest = std::max({std::fabs(left), std::fabs(top), std::fabs(right), std::fabs(bottom)});
        if (largest <= 2.0F) {
            left *= static_cast<float>(network.width); right *= static_cast<float>(network.width);
            top *= static_cast<float>(network.height); bottom *= static_cast<float>(network.height);
        }
        left = std::clamp(left, 0.0F, static_cast<float>(network.width));
        right = std::clamp(right, 0.0F, static_cast<float>(network.width));
        top = std::clamp(top, 0.0F, static_cast<float>(network.height));
        bottom = std::clamp(bottom, 0.0F, static_cast<float>(network.height));
        if (right <= left || bottom <= top) continue;
        NvDsInferObjectDetectionInfo object{};
        object.classId = static_cast<unsigned int>(class_id); object.detectionConfidence = score;
        object.left = left; object.top = top; object.width = right - left; object.height = bottom - top;
        objects.push_back(object);
    }
    return true;
}

bool rockchip_head_shape(const NvDsInferLayerInfo& layer, RockchipHead& head) {
    if (layer.buffer == nullptr || (layer.dataType != FLOAT && layer.dataType != HALF)) {
        return false;
    }
    std::vector<std::size_t> dimensions;
    dimensions.reserve(layer.inferDims.numDims);
    for (unsigned int index = 0; index < layer.inferDims.numDims; ++index) {
        const int value = layer.inferDims.d[index];
        if (value <= 0) {
            return false;
        }
        dimensions.push_back(static_cast<std::size_t>(value));
    }
    if (dimensions.size() == 4U && dimensions.front() == 1U) {
        dimensions.erase(dimensions.begin());
    }
    if (dimensions.size() != 3U || dimensions[1] != dimensions[2]) {
        return false;
    }
    head = RockchipHead{&layer, dimensions[0], dimensions[1], dimensions[2]};
    return true;
}

float rockchip_value(
    const RockchipHead& head,
    std::size_t channel,
    std::size_t y,
    std::size_t x) {
    const std::size_t index = (channel * head.height + y) * head.width + x;
    return layer_value(*head.layer, index);
}

bool parse_rockchip_yolov5(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects,
    std::size_t& input_candidates) {
    if (layers.size() != 3U || params.numClassesConfigured == 0U) {
        return false;
    }
    std::vector<RockchipHead> heads(3U);
    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (!rockchip_head_shape(layers[index], heads[index])) {
            return false;
        }
    }
    std::sort(heads.begin(), heads.end(), [](const RockchipHead& left, const RockchipHead& right) {
        return left.width > right.width;
    });
    const std::size_t expected_channels = kRockchipAnchorCount
        * (params.numClassesConfigured + 5U);
    input_candidates = 0U;
    for (std::size_t scale = 0; scale < heads.size(); ++scale) {
        const auto& head = heads[scale];
        const std::size_t expected_stride = 8U << scale;
        if (head.channels != expected_channels
            || head.width * expected_stride != network.width
            || head.height * expected_stride != network.height) {
            return false;
        }
        input_candidates += kRockchipAnchorCount * head.width * head.height;
    }

    objects.reserve(objects.size() + std::min(input_candidates, kInitialObjectCapacity));
    const std::size_t values_per_anchor = params.numClassesConfigured + 5U;
    for (std::size_t scale = 0; scale < heads.size(); ++scale) {
        const auto& head = heads[scale];
        const float stride_x = static_cast<float>(network.width) / static_cast<float>(head.width);
        const float stride_y = static_cast<float>(network.height) / static_cast<float>(head.height);
        for (std::size_t anchor = 0; anchor < kRockchipAnchorCount; ++anchor) {
            const std::size_t channel_base = anchor * values_per_anchor;
            for (std::size_t y = 0; y < head.height; ++y) {
                for (std::size_t x = 0; x < head.width; ++x) {
                    const float objectness = rockchip_value(head, channel_base + 4U, y, x);
                    if (!std::isfinite(objectness) || objectness <= 0.0F) {
                        continue;
                    }
                    std::size_t best_class = 0U;
                    float best_score = -std::numeric_limits<float>::infinity();
                    for (std::size_t class_id = 0; class_id < params.numClassesConfigured; ++class_id) {
                        const float class_probability = rockchip_value(
                            head, channel_base + 5U + class_id, y, x);
                        const float score = objectness * class_probability;
                        if (score > best_score) {
                            best_score = score;
                            best_class = class_id;
                        }
                    }
                    if (!std::isfinite(best_score)
                        || best_score < threshold_for_class(params, best_class)) {
                        continue;
                    }
                    const float raw_x = rockchip_value(head, channel_base, y, x);
                    const float raw_y = rockchip_value(head, channel_base + 1U, y, x);
                    const float raw_w = rockchip_value(head, channel_base + 2U, y, x);
                    const float raw_h = rockchip_value(head, channel_base + 3U, y, x);
                    if (!std::isfinite(raw_x) || !std::isfinite(raw_y)
                        || !std::isfinite(raw_w) || !std::isfinite(raw_h)) {
                        continue;
                    }
                    const float cx = (raw_x * 2.0F - 0.5F + static_cast<float>(x)) * stride_x;
                    const float cy = (raw_y * 2.0F - 0.5F + static_cast<float>(y)) * stride_y;
                    const float scaled_w = raw_w * 2.0F;
                    const float scaled_h = raw_h * 2.0F;
                    const float width = scaled_w * scaled_w * kRockchipYoloV5Anchors[scale][anchor][0];
                    const float height = scaled_h * scaled_h * kRockchipYoloV5Anchors[scale][anchor][1];
                    const float left = std::max(0.0F, cx - width * 0.5F);
                    const float top = std::max(0.0F, cy - height * 0.5F);
                    const float right = std::min(static_cast<float>(network.width), cx + width * 0.5F);
                    const float bottom = std::min(static_cast<float>(network.height), cy + height * 0.5F);
                    if (right <= left || bottom <= top) {
                        continue;
                    }
                    NvDsInferObjectDetectionInfo object{};
                    object.classId = static_cast<unsigned int>(best_class);
                    object.detectionConfidence = best_score;
                    object.left = left;
                    object.top = top;
                    object.width = right - left;
                    object.height = bottom - top;
                    objects.push_back(object);
                }
            }
        }
    }
    return true;
}

}  // namespace

extern "C" bool NvDsInferParseNovaSight(
    const std::vector<NvDsInferLayerInfo>& output_layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects) {
    const auto started = std::chrono::steady_clock::now();
    g_decode_calls.fetch_add(1U, std::memory_order_relaxed);
    const auto fail = [&](std::uint64_t error_code) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
        g_parse_failures.fetch_add(1U, std::memory_order_relaxed);
        g_last_error_code.store(error_code, std::memory_order_relaxed);
        g_last_decode_ns.store(
            static_cast<std::uint64_t>(elapsed.count()),
            std::memory_order_relaxed);
        return false;
    };
    if (output_layers.size() == 3U) {
        std::size_t input_candidates = 0U;
        if (!parse_rockchip_yolov5(
                output_layers, network, params, objects, input_candidates)) {
            // DeepStream versions in the field may abort after a custom parser
            // returns false. Keep the error in telemetry, but return an empty
            // detection batch so a malformed contract cannot crash the process.
            fail(5U);
            return true;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
        g_last_error_code.store(0U, std::memory_order_relaxed);
        g_last_decode_ns.store(
            static_cast<std::uint64_t>(elapsed.count()), std::memory_order_relaxed);
        g_last_input_candidates.store(input_candidates, std::memory_order_relaxed);
        g_last_output_candidates.store(objects.size(), std::memory_order_relaxed);
        return true;
    }
    if (output_layers.size() == 4U) {
        std::size_t input_candidates = 0U;
        if (!parse_efficient_nms(
                output_layers, network, params, objects, input_candidates)) {
            fail(4U);
            return true;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
        g_last_error_code.store(0U, std::memory_order_relaxed);
        g_last_decode_ns.store(
            static_cast<std::uint64_t>(elapsed.count()), std::memory_order_relaxed);
        g_last_input_candidates.store(input_candidates, std::memory_order_relaxed);
        g_last_output_candidates.store(objects.size(), std::memory_order_relaxed);
        return true;
    }
    if (output_layers.size() != 1U || output_layers.front().buffer == nullptr) {
        fail(1U);
        return true;
    }
    const NvDsInferLayerInfo& layer = output_layers.front();
    if (layer.dataType != FLOAT && layer.dataType != HALF) {
        fail(2U);
        return true;
    }
    const std::size_t class_count = params.numClassesConfigured;
    // Prefer the explicit decoded-box contract when the tensor has six
    // columns. Treating it as raw YOLO cx/cy/w/h is incorrect and can feed
    // invalid geometry into downstream DeepStream code.
    {
        std::vector<std::size_t> dims;
        for (unsigned int i = 0; i < layer.inferDims.numDims; ++i) {
            if (layer.inferDims.d[i] > 0) dims.push_back(static_cast<std::size_t>(layer.inferDims.d[i]));
        }
        if (!dims.empty() && dims.back() == 6U) {
            std::size_t input_candidates = 0U;
            if (!parse_decoded_boxes6(layer, network, params, objects, input_candidates)) {
                fail(6U);
                return true;
            }
            g_last_error_code.store(0U, std::memory_order_relaxed);
            g_last_input_candidates.store(input_candidates, std::memory_order_relaxed);
            g_last_output_candidates.store(objects.size(), std::memory_order_relaxed);
            g_last_decode_ns.store(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()),
                std::memory_order_relaxed);
            return true;
        }
    }
    std::size_t channels = 0;
    std::size_t candidates = 0;
    bool channels_first = false;
    if (!output_shape(layer, class_count, channels, candidates, channels_first)) {
        fail(3U);
        return true;
    }
    const bool has_objectness = channels == class_count + 5U;
    const std::size_t class_offset = has_objectness ? 5U : 4U;
    // Raw YOLO outputs commonly contain thousands of candidates, while only a
    // small post-threshold subset becomes NvDs objects. Reserving for every raw
    // candidate caused a large allocation on every inference callback.
    objects.reserve(objects.size() + std::min(candidates, kInitialObjectCapacity));
    for (std::size_t candidate = 0; candidate < candidates; ++candidate) {
        const float cx = prediction_value(
            layer, candidate, 0U, channels, candidates, channels_first);
        const float cy = prediction_value(
            layer, candidate, 1U, channels, candidates, channels_first);
        const float width = prediction_value(
            layer, candidate, 2U, channels, candidates, channels_first);
        const float height = prediction_value(
            layer, candidate, 3U, channels, candidates, channels_first);
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(width)
            || !std::isfinite(height) || width <= 0.0F || height <= 0.0F) {
            continue;
        }
        const float objectness = has_objectness
            ? prediction_value(layer, candidate, 4U, channels, candidates, channels_first)
            : 1.0F;
        std::size_t best_class = 0;
        float best_score = -std::numeric_limits<float>::infinity();
        for (std::size_t class_id = 0; class_id < class_count; ++class_id) {
            const float class_score = prediction_value(
                layer,
                candidate,
                class_offset + class_id,
                channels,
                candidates,
                channels_first);
            const float score = objectness * class_score;
            if (score > best_score) {
                best_score = score;
                best_class = class_id;
            }
        }
        if (!std::isfinite(best_score) || best_score < threshold_for_class(params, best_class)) {
            continue;
        }
        const float left = std::max(0.0F, cx - width * 0.5F);
        const float top = std::max(0.0F, cy - height * 0.5F);
        const float right = std::min(static_cast<float>(network.width), cx + width * 0.5F);
        const float bottom = std::min(static_cast<float>(network.height), cy + height * 0.5F);
        if (right <= left || bottom <= top) {
            continue;
        }
        NvDsInferObjectDetectionInfo object{};
        object.classId = static_cast<unsigned int>(best_class);
        object.detectionConfidence = best_score;
        object.left = left;
        object.top = top;
        object.width = right - left;
        object.height = bottom - top;
        objects.push_back(object);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
    g_last_error_code.store(0U, std::memory_order_relaxed);
    g_last_decode_ns.store(static_cast<std::uint64_t>(elapsed.count()), std::memory_order_relaxed);
    g_last_input_candidates.store(candidates, std::memory_order_relaxed);
    g_last_output_candidates.store(objects.size(), std::memory_order_relaxed);
    return true;
}

extern "C" std::uint64_t novasight_parser_decode_calls() {
    return g_decode_calls.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t novasight_parser_last_decode_ns() {
    return g_last_decode_ns.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t novasight_parser_parse_failures() {
    return g_parse_failures.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t novasight_parser_last_error_code() {
    return g_last_error_code.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t novasight_parser_last_input_candidates() {
    return g_last_input_candidates.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t novasight_parser_last_output_candidates() {
    return g_last_output_candidates.load(std::memory_order_relaxed);
}

extern "C" void novasight_parser_reset_telemetry() {
    g_decode_calls.store(0U, std::memory_order_relaxed);
    g_parse_failures.store(0U, std::memory_order_relaxed);
    g_last_error_code.store(0U, std::memory_order_relaxed);
    g_last_decode_ns.store(0U, std::memory_order_relaxed);
    g_last_input_candidates.store(0U, std::memory_order_relaxed);
    g_last_output_candidates.store(0U, std::memory_order_relaxed);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseNovaSight);
