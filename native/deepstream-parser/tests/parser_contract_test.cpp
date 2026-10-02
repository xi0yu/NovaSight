#include <nvdsinfer_custom_impl.h>

#include <cmath>
#include <cstdint>
#include <vector>

extern "C" bool NvDsInferParseNovaSight(
    const std::vector<NvDsInferLayerInfo>& output_layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects);
extern "C" bool NvDsInferParseNovaSightRaw(
    const std::vector<NvDsInferLayerInfo>& output_layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects);
extern "C" bool NvDsInferParseNovaSightDecodedNms(
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

bool test_ambiguous_six_column_contracts_use_explicit_entrypoints() {
    // The exact same [1,N,6] shape can mean one-class raw YOLOv5 or decoded NMS.
    // The manifest-selected parser entrypoint, not a shape guess, owns the meaning.
    float raw[] = {100.0F, 120.0F, 40.0F, 20.0F, 0.9F, 0.8F};
    NvDsInferLayerInfo raw_layer{"output0", raw, FLOAT, dims({1, 1, 6})};
    NvDsInferNetworkInfo network{640U, 640U};
    NvDsInferParseDetectionParams one_class{};
    one_class.numClassesConfigured = 1U;
    one_class.perClassPreclusterThreshold = {0.25F};
    std::vector<NvDsInferObjectDetectionInfo> raw_objects;
    if (!NvDsInferParseNovaSightRaw({raw_layer}, network, one_class, raw_objects)
        || raw_objects.size() != 1U) {
        return false;
    }
    if (std::fabs(raw_objects.front().left - 80.0F) > 0.01F
        || std::fabs(raw_objects.front().top - 110.0F) > 0.01F
        || std::fabs(raw_objects.front().detectionConfidence - 0.72F) > 0.001F) {
        return false;
    }

    float decoded[] = {80.0F, 110.0F, 120.0F, 130.0F, 0.9F, 0.0F};
    NvDsInferLayerInfo decoded_layer{"detections", decoded, FLOAT, dims({1, 1, 6})};
    std::vector<NvDsInferObjectDetectionInfo> decoded_objects;
    return NvDsInferParseNovaSightDecodedNms(
               {decoded_layer}, network, one_class, decoded_objects)
        && decoded_objects.size() == 1U
        && std::fabs(decoded_objects.front().left - 80.0F) <= 0.01F
        && std::fabs(decoded_objects.front().width - 40.0F) <= 0.01F;
}

}  // namespace

int main() {
    if (!test_efficient_nms()) {
        return 1;
    }
    if (!test_rockchip_yolov5()) {
        return 2;
    }
    if (!test_ambiguous_six_column_contracts_use_explicit_entrypoints()) {
        return 3;
    }
    return 0;
}
