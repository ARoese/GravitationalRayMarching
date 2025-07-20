//
// Created by atomr on 7/19/2025.
//

#include "Renderer.cuh"
#include "api/C_Rendering.h"

extern "C" {
    COMPAT_PTRTYPE(Texture) render_gpu(COMPAT_PTRTYPE(Scene) scene, const D_RenderConfig d_config){
        const auto* scene_p = static_cast<Scene*>(scene);
        Renderer renderer;
        const CudaContext cudaContext;
        const RenderConfig config(d_config);
        Texture result = renderer.renderGPU(*scene_p, config, cudaContext);
        return new Texture(std::move(result));
    }

    COMPAT_PTRTYPE(Texture) render_cpu(COMPAT_PTRTYPE(Scene) scene, const D_RenderConfig d_config){
        auto* scene_p = static_cast<Scene*>(scene);
        Renderer renderer;
        Texture result = renderer.renderCPU(*scene_p, d_config);
        return new Texture(std::move(result));
    }
}