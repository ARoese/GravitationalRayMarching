#ifndef CAMERA_H
#define CAMERA_H
#include <vector_types.h>
#include "api/UniversalConstants.h"
#include "api/D_Camera.h"

class Camera {
    public:
    float2 fov = {90*RAD2DEG, 90*RAD2DEG};
    float3 camPos = {0,0,0};
    float3 camRot = {0,0,0};
    __host__ Camera(float2 fov, float3 camPos, float3 camRot);
    __host__ Camera(D_Camera o) :
        fov({o.fov.x, o.fov.y}),
        camPos({o.camPos.x, o.camPos.y, o.camPos.z,}),
        camRot({o.camRot.x, o.camRot.y, o.camRot.z,}) {}
};

#endif