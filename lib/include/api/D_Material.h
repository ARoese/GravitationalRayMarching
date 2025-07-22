//
// Created by atomr on 7/17/2025.
//

#ifndef MATERIAL_H
#define MATERIAL_H
#include "compat_wrappers.h"
#include "D_VectorTypes.h"

#ifdef __cplusplus
    extern "C" {
#endif
        typedef struct {
            enum Type {
                FLAT,
                TEXTURE
            } type;

            union {
                d_uchar3 color;
                const char* texture_path;
            };
        } D_Material;

        COMPAT_FOR(Material)
#ifdef __cplusplus
    }
#endif

#endif //MATERIAL_H
