//
// Created by atomr on 7/19/2025.
//

#ifndef RENDERING_H
#define RENDERING_H

#include "compat_wrappers.h"
#include "C_Texture.h"
#include "D_Scene.h"
#include "D_RenderConfig.h"
#include "C_CancellationToken.h"

#ifdef __cplusplus
    extern "C" {
#endif
        COMPAT_PTRTYPE(Texture) render_gpu(COMPAT_PTRTYPE(Scene), D_RenderConfig);
        COMPAT_PTRTYPE(Texture) render_cpu(COMPAT_PTRTYPE(Scene), D_RenderConfig);

        // result may be nullptr if CancellationToken was canceled during evaluation
        COMPAT_PTRTYPE(Texture) render_gpu_cancelable(COMPAT_PTRTYPE(Scene), D_RenderConfig, COMPAT_PTRTYPE(CancellationToken));
        // result may be nullptr if CancellationToken was canceled during evaluation
        COMPAT_PTRTYPE(Texture) render_cpu_cancelable(COMPAT_PTRTYPE(Scene), D_RenderConfig, COMPAT_PTRTYPE(CancellationToken));


#ifdef __cplusplus
    }
#endif

#endif //RENDERING_H
