//
// Created by atomr on 7/17/2025.
//

#ifndef D_SCENE_H
#define D_SCENE_H
#include "D_Body.h"
#include "D_Camera.h"
#include "D_Material.h"
#include "UniversalConstants.h"
#ifdef __cplusplus
    extern "C" {
#endif

        typedef struct {
            D_Camera cam;
            D_Body* bodies;
            size_t n_bodies;
            UniversalConstants constants;
            D_Material nohit;
        } D_Scene;
        COMPAT_FOR(Scene)

#ifdef __cplusplus
        }
#endif

#endif //D_SCENE_H
