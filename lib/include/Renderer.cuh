#ifndef __RENDERING_H__
#define __RENDERING_H__

#include <optional>
#include <vector_types.h>

#include "Body.cuh"

#include "CudaContext.cuh"
#include "Scene.cuh"
#include "RenderConfig.cuh"

struct CancellationToken;

class Renderer {
public:
    std::optional<FrameBuffer> renderGPU(const Scene& scene, const RenderConfig& config, const CudaContext& cudaContext, const CancellationToken &ct);
    std::optional<FrameBuffer> renderCPU(const Scene& scene, const RenderConfig& config, const CancellationToken &ct);

    FrameBuffer renderGPU(const Scene &scene, const RenderConfig &config, const CudaContext &cudaContext);
    FrameBuffer renderCPU(const Scene &scene, const RenderConfig &config);
};


#endif