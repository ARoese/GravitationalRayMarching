#include "Body.cuh"

Body::Body(float rad, float mass, float3 position, float3 rotation, Material mat)
    : mass(mass), radius(rad), position(position), rotation(rotation), material(std::move(mat))
    {}

void Body::toDeviceImpl(Body* deviceLocation) const {
    simpleCopyToDevice(deviceLocation);
    material.toDevice(&deviceLocation->material);
}

void Body::fromDeviceImpl(Body* deviceLocation) {
    simpleCopyFromDevice(deviceLocation);
    material.fromDevice(&deviceLocation->material);
}