//
// Created by atomr on 7/17/2025.
//

#ifndef D_CAMERA_H
#define D_CAMERA_H
#include "compat_wrappers.h"
#include "D_VectorTypes.h"

#ifdef __cplusplus
    extern "C" {
#endif

        typedef struct {
            d_float2 fov;
            d_float3 camPos;
            d_float3 camRot;
            d_uint2 resolution;
        } D_Camera;
        COMPAT_FOR(Camera)

#ifdef __cplusplus
    }
#endif

#endif //D_CAMERA_H
