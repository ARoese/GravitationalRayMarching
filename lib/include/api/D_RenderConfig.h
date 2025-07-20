//
// Created by atomr on 7/20/2025.
//

#ifndef RENDEROPTIONS_H
#define RENDEROPTIONS_H

#include "D_VectorTypes.h"

#ifdef __cplusplus
    extern "C" {
#endif

        typedef struct {
            // how many steps should the raymarch take from the camera?
            int marchSteps;
            // how much time should pass in each step?
            // (lower means higher step resolution, but needs more steps to cover the same distance)
            float marchStepDeltaTime;
        } D_MarchConfig;

        const D_MarchConfig default_MarchConfig = {
            10000,
            0.005
        };

        typedef struct {
            d_uint2 resolution;
            D_MarchConfig marchConfig;
        } D_RenderConfig;

#ifdef __cplusplus
        }
#endif

#endif //RENDEROPTIONS_H
