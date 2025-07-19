#include "Body.cuh"

Body::Body(float rad, float mass, float3 position, float3 rotation, Material mat)
    : radius(rad), mass(mass), position(position), rotation(rotation), material(std::move(mat))
    {}

Body::Body() : radius(0.0f), mass(0.0f), position({0,0,0}), rotation({0,0,0}), material(Material({0,0,0})){}

void Body::toDeviceImpl(Body* deviceLocation) const {
    simpleCopyToDevice(deviceLocation);
    material.toDevice(&deviceLocation->material);
}

void Body::fromDeviceImpl(Body* deviceLocation) {
    simpleCopyFromDevice(deviceLocation);
    material.fromDevice(&deviceLocation->material);
}