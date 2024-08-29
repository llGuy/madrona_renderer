#pragma once

#include <string>

namespace run {

enum class ColorType {
    RGB, Depth
};
    
struct DumpInfo {
    std::string outputPath;

    // Pointer to CUDA memory containing the images.
    void *gpuTensor;

    // We will calculate what the best resolution is for this output.
    uint32_t numImages;

    // Resolution of each individual imagea
    uint32_t imageResolution;

    ColorType colorType;
};

void dumpTiledImage(const DumpInfo &info);

}
