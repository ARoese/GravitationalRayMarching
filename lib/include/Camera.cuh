#ifndef CAMERA_H
#define CAMERA_H
#include <vector_types.h>

#include "Transferable.cuh"
#include "universalConstants.hpp"

class Camera {
    public:
    float2 fov = {90*RAD2DEG, 90*RAD2DEG};
    float3 camPos = {0,0,0};
    float3 camRot = {0,0,0};
    uint2 resolution = {255,255};
    __host__ Camera(float2 fov, float3 camPos, float3 camRot, uint2 resolution);
};

#endif