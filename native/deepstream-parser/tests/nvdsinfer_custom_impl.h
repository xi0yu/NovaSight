#pragma once

#include <vector>

enum NvDsInferDataType { FLOAT, HALF, INT8, INT32 };

struct NvDsInferDims {
    unsigned int numDims = 0;
    int d[8]{};
};

struct NvDsInferLayerInfo {
    void* buffer = nullptr;
    NvDsInferDataType dataType = FLOAT;
    NvDsInferDims inferDims{};
};

struct NvDsInferNetworkInfo {
    unsigned int width = 0;
    unsigned int height = 0;
};

struct NvDsInferParseDetectionParams {
    unsigned int numClassesConfigured = 0;
    std::vector<float> perClassPreclusterThreshold;
};

struct NvDsInferObjectDetectionInfo {
    unsigned int classId = 0;
    float detectionConfidence = 0.0F;
    float left = 0.0F;
    float top = 0.0F;
    float width = 0.0F;
    float height = 0.0F;
};

#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(name) static_assert(true, #name)
