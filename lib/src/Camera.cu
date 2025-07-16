#include "Camera.cuh"


__host__ Camera::Camera(float2 fov, float3 camPos, float3 camRot, uint2 resolution)
    : fov(fov), camPos(camPos), camRot(camRot), resolution(resolution)
    {}