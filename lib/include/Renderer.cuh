#ifndef __RENDERING_H__
#define __RENDERING_H__

#include <vector_types.h>

#include "Body.cuh"

#include "CudaContext.cuh"
#include "Scene.cuh"

class Renderer {
public:
    FrameBuffer renderGPU(Scene& scene, CudaContext& cudaContext);
    FrameBuffer renderCPU(Scene& scene);
};


#endif