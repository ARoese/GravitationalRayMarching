//
// Created by atomr on 7/19/2025.
//

#ifndef RENDERING_H
#define RENDERING_H

#include "compat_wrappers.h"
#include "C_Texture.h"
#include "D_Scene.h"

#ifdef __cplusplus
    extern "C" {
#endif

        COMPAT_PTRTYPE(Texture) render_gpu(COMPAT_PTRTYPE(Scene));
        COMPAT_PTRTYPE(Texture) render_cpu(COMPAT_PTRTYPE(Scene));


#ifdef __cplusplus
    }
#endif

#endif //RENDERING_H
