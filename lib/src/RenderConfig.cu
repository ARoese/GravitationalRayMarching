//
// Created by atomr on 7/20/2025.
//

#include "RenderConfig.cuh"

#include "api/compat_vector_types.cuh"

RenderConfig::RenderConfig(const D_RenderConfig& config):
        resolution(fromCompat(config.resolution)),
        marchConfig(config.marchConfig) {}
