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

}  // namespace

int main() {
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

    if (!NvDsInferParseNovaSight(layers, network, params, objects)) {
        return 1;
    }
    if (objects.size() != 1U || objects.front().classId != 1U) {
        return 2;
    }
    const auto& object = objects.front();
    if (std::fabs(object.left - 64.0F) > 0.01F
        || std::fabs(object.top - 128.0F) > 0.01F
        || std::fabs(object.width - 128.0F) > 0.01F
        || std::fabs(object.height - 128.0F) > 0.01F) {
        return 3;
    }
    return 0;
}
