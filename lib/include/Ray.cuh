#ifndef RAY_H
#define RAY_H
#include <vector_types.h>
#include "Body.cuh"

class Ray{
    public:
    float3 position;
    float3 direction;
    __device__ __host__ Ray(float3 position, float3 direction);
    //steps the ray in its direction and updates direction based on gravity
    __device__ __host__ void step(float timestep, const Buffer<Body>& bodies, const UniversalConstants& c);
    private:
    __device__ __host__ float3 getAccelerationTo(const Body& b, const UniversalConstants& constants) const;
};
#endif