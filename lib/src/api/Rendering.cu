//
// Created by atomr on 7/19/2025.
//

#include "Renderer.cuh"
#include "api/C_Rendering.h"

extern "C" {
    COMPAT_PTRTYPE(Texture) render_gpu_cancelable(COMPAT_PTRTYPE(Scene) scene, D_RenderConfig d_config, COMPAT_PTRTYPE(CancellationToken) ct) {
        const auto* scene_p = static_cast<Scene*>(scene);
        auto* ct_p = static_cast<CancellationToken*>(ct);
        Renderer renderer;
        const CudaContext cudaContext;
        const RenderConfig config(d_config);
        auto result = renderer.renderGPU(*scene_p, config, cudaContext, *ct_p);
        return result ? new Texture(std::move(*result)) : nullptr;
    }
    COMPAT_PTRTYPE(Texture) render_cpu_cancelable(COMPAT_PTRTYPE(Scene) scene, D_RenderConfig d_config, COMPAT_PTRTYPE(CancellationToken) ct) {
        auto* scene_p = static_cast<Scene*>(scene);
        auto* ct_p = static_cast<CancellationToken*>(ct);
        Renderer renderer;
        const RenderConfig config(d_config);
        auto result = renderer.renderCPU(*scene_p, config, *ct_p);
        return result ? new Texture(std::move(*result)) : nullptr;
    }

    COMPAT_PTRTYPE(Texture) render_gpu(COMPAT_PTRTYPE(Scene) scene, const D_RenderConfig d_config){
        const auto* scene_p = static_cast<Scene*>(scene);
        Renderer renderer;
        const CudaContext cudaContext;
        const RenderConfig config(d_config);
        auto result = renderer.renderGPU(*scene_p, config, cudaContext);
        return new Texture(std::move(result));
    }

    COMPAT_PTRTYPE(Texture) render_cpu(COMPAT_PTRTYPE(Scene) scene, const D_RenderConfig d_config){
        auto* scene_p = static_cast<Scene*>(scene);
        Renderer renderer;
        const RenderConfig config(d_config);
        auto result = renderer.renderCPU(*scene_p, config);
        return new Texture(std::move(result));
    }
}