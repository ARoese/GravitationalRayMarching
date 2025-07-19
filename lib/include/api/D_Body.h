//
// Created by atomr on 7/17/2025.
//

#ifndef D_BODY_H
#define D_BODY_H
#include "compat_wrappers.h"
#include "D_VectorTypes.h"
#include "D_Material.h"
#include "UniversalConstants.h"

#ifdef __cplusplus
    extern "C" {
#endif

        typedef struct {
            float radius;
            float mass;
            d_float3 position;
            d_float3 rotation;
            D_Material material;
        } D_Body;

        float getSchwarzschildRadiusForMass(float mass, UniversalConstants c);

        COMPAT_FOR(Body)


#ifdef __cplusplus
    }
#endif

#endif //D_BODY_H
