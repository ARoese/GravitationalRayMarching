#include "Ray.cuh"
#include "cutil_math.cuh"
#include "../include/api/UniversalConstants.h"

Ray::Ray(float3 position, float3 direction)
    : position(position), direction(direction)
    {}

void Ray::step(float timestep, const Buffer<Body>& bodies, const UniversalConstants& c){
    direction *= c.C;
    position += direction*timestep;
    for(int i = 0; i < bodies.elems; i++){
        direction += getAccelerationTo(bodies[i], c)*timestep;
    }
    direction = normalize(direction);
}

float3 Ray::getAccelerationTo(const Body& b, const UniversalConstants& c) const{
    float3 directionToBody = b.position - position;
    float amplitude = ((c.G*b.mass)/length(directionToBody));
    return normalize(directionToBody)*amplitude;
}