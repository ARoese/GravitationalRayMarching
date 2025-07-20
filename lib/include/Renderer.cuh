#ifndef __RENDERING_H__
#define __RENDERING_H__

#include <vector_types.h>

#include "Body.cuh"

#include "CudaContext.cuh"
#include "Scene.cuh"
#include "RenderConfig.cuh"

class Renderer {
public:
    FrameBuffer renderGPU(const Scene &scene, const RenderConfig &config, const CudaContext &cudaContext);
    FrameBuffer renderCPU(const Scene &scene, const RenderConfig &config);
};


#endif