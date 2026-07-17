#include <nvdsinfer_custom_impl.h>

#include <cmath>
#include <cstdint>
#include <vector>

extern "C" bool NvDsInferParseNovaSight(
    const std::vector<NvDsInferLayerInfo>& output_layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects);

namespace {

NvDsInferDims dims(std::initializer_list<int> values) {
    NvDsInferDims result{};
    result.numDims = static_cast<unsigned int>(values.size());
    unsigned int index = 0U;
    for (const int value : values) {
        result.d[index++] = value;
    }
    return result;
}

bool test_efficient_nms() {
    std::int32_t count[] = {2};
    float boxes[] = {
        0.10F, 0.20F, 0.30F, 0.40F,
        0.40F, 0.40F, 0.50F, 0.50F,
    };
    float scores[] = {0.90F, 0.10F};
    std::int32_t classes[] = {1, 0};
    std::vector<NvDsInferLayerInfo> layers{
        {"num_dets", count, INT32, dims({1, 1})},
        {"det_boxes", boxes, FLOAT, dims({1, 2, 4})},
        {"det_scores", scores, FLOAT, dims({1, 2})},
        {"det_classes", classes, INT32, dims({1, 2})},
    };
    NvDsInferNetworkInfo network{640U, 640U};
    NvDsInferParseDetectionParams params{};
    params.numClassesConfigured = 2U;
    params.perClassPreclusterThreshold = {0.25F, 0.25F};
    std::vector<NvDsInferObjectDetectionInfo> objects;
    if (!NvDsInferParseNovaSight(layers, network, params, objects)
        || objects.size() != 1U || objects.front().classId != 1U) {
        return false;
    }
    const auto& object = objects.front();
    return std::fabs(object.left - 64.0F) <= 0.01F
        && std::fabs(object.top - 128.0F) <= 0.01F
        && std::fabs(object.width - 128.0F) <= 0.01F
        && std::fabs(object.height - 128.0F) <= 0.01F;
}

void set_nchw(
    std::vector<float>& tensor,
    std::size_t channels,
    std::size_t height,
    std::size_t width,
    std::size_t channel,
    std::size_t y,
    std::size_t x,
    float value) {
    (void)channels;
    tensor[(channel * height + y) * width + x] = value;
}

bool test_rockchip_yolov5() {
    constexpr std::size_t channels = 21U;
    std::vector<float> head40(channels * 40U * 40U, 0.0F);
    std::vector<float> head20(channels * 20U * 20U, 0.0F);
    std::vector<float> head10(channels * 10U * 10U, 0.0F);
    set_nchw(head40, channels, 40U, 40U, 0U, 2U, 1U, 0.5F);
    set_nchw(head40, channels, 40U, 40U, 1U, 2U, 1U, 0.5F);
    set_nchw(head40, channels, 40U, 40U, 2U, 2U, 1U, 0.5F);
    set_nchw(head40, channels, 40U, 40U, 3U, 2U, 1U, 0.5F);
    set_nchw(head40, channels, 40U, 40U, 4U, 2U, 1U, 0.9F);
    set_nchw(head40, channels, 40U, 40U, 5U, 2U, 1U, 0.1F);
    set_nchw(head40, channels, 40U, 40U, 6U, 2U, 1U, 0.8F);

    // Deliberately shuffled to prove that binding order does not control anchor scale.
    std::vector<NvDsInferLayerInfo> layers{
        {"288", head10.data(), FLOAT, dims({1, 21, 10, 10})},
        {"output0", head40.data(), FLOAT, dims({1, 21, 40, 40})},
        {"286", head20.data(), FLOAT, dims({1, 21, 20, 20})},
    };
    NvDsInferNetworkInfo network{320U, 320U};
    NvDsInferParseDetectionParams params{};
    params.numClassesConfigured = 2U;
    params.perClassPreclusterThreshold = {0.25F, 0.25F};
    std::vector<NvDsInferObjectDetectionInfo> objects;
    if (!NvDsInferParseNovaSight(layers, network, params, objects)
        || objects.size() != 1U || objects.front().classId != 1U) {
        return false;
    }
    const auto& object = objects.front();
    return std::fabs(object.detectionConfidence - 0.72F) <= 0.001F
        && std::fabs(object.left - 7.0F) <= 0.01F
        && std::fabs(object.top - 13.5F) <= 0.01F
        && std::fabs(object.width - 10.0F) <= 0.01F
        && std::fabs(object.height - 13.0F) <= 0.01F;
}

}  // namespace

int main() {
    if (!test_efficient_nms()) {
        return 1;
    }
    if (!test_rockchip_yolov5()) {
        return 2;
    }
    return 0;
}
