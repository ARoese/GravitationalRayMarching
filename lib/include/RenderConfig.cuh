//
// Created by atomr on 7/20/2025.
//

#ifndef RENDERCONFIG_CUH
#define RENDERCONFIG_CUH

#include "Transferable.cuh"
#include "api/D_RenderConfig.h"

struct RenderConfig : Transferable<RenderConfig> {
    uint2 resolution = {256, 256};
    D_MarchConfig marchConfig = default_MarchConfig;

    RenderConfig(const D_RenderConfig& config);
    RenderConfig(uint2 resolution, D_MarchConfig marchConfig) :
        resolution(resolution),
        marchConfig(marchConfig) {}
};

#endif //RENDERCONFIG_CUH
