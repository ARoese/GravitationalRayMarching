#ifndef BODY_H
#define BODY_H
#include <vector_types.h>

#include "Material.cuh"
#include "Transferable.cuh"
#include "universalConstants.hpp"

class Body: public Transferable<Body> {
    public:
    float radius;
    float mass;
    float3 position;
    float3 rotation;
    Material material;
    Body(Body&& o) noexcept:
        radius(o.radius),
        mass(o.mass),
        position(o.position),
        rotation(o.rotation),
        material(std::move(o.material)) {}
    __host__ Body(float radius, float mass, float3 position, float3 rotation, Material material);
    __host__ __device__ static float getSchwarzschildRadiusForMass(float mass, UniversalConstants c){
        return (2*c.G*mass)/c.C;
    }

    __host__ void toDeviceImpl(Body* deviceLocation) const;
    __host__ void fromDeviceImpl(Body* deviceLocation);
};
#endif