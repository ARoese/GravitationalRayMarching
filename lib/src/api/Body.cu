//
// Created by atomr on 7/18/2025.
//

#include "Body.cuh"

#include "compat_impl.cuh"
#include "api/D_Body.h"
#include "api/compat_wrappers.h"
#include "compat_vector_types.cuh"

extern "C" {
    COMPAT_CONSTRUCTOR(Body) {
        return new Body(
            dc.radius,
            dc.mass,
            fromCompat(dc.position),
            fromCompat(dc.rotation),
            dc.material
            );
    }

    float getSchwarzschildRadiusForMass(float mass, UniversalConstants c) {
        return Body::getSchwarzschildRadiusForMass(mass, c);
    }

    COMPAT_DESTRUCTOR_IMPL(Body)
}

