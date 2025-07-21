#include "Renderer.cuh"

#include <algorithm>
#include <execution>
#include <format>
#include <iostream>
#include <vector>

#include "CancellationToken.cuh"
#include "cutil_math.cuh"
#include "cmatrix.cuh"

#include "RenderConfig.cuh"
#include "Ray.cuh"
#include "loadImage.hpp"

constexpr uchar3 debug_purple = {255, 0, 255};

__device__ __host__ float2 uvMapSphere(const float3 rayPos, const float3 bodyPos, const float3 bodyRot){
    //https://en.wikipedia.org/wiki/UV_mapping
    float3 d = normalize(rayPos - bodyPos);

    d = float3x3::Rx(bodyRot.x)*(float3x3::Ry(bodyRot.y)*(float3x3::Rz(bodyRot.z)*d));

    float u = 0.5 + atan2(d.z, d.x)/(2*M_PI);
    float v = 0.5 + asin(d.y)/M_PI;

    return {u,v};
}

__device__ __host__ uchar3 renderPixel(const Scene& scene, const RenderConfig& config, const CancellationToken& ct, float2 screenSpace){
    if (ct.cancelled) {
        return debug_purple;
    }
    //angular screenspace coordinates from [-fov.x, fov.x] and so on
    const float2 screenAngle = (screenSpace-0.5) * scene.cam.fov;

    float3 rayDir = {1,0,0};
    //local rotation
    rayDir = float3x3::Ry(screenAngle.y)
                    *(float3x3::Rz(screenAngle.x)*rayDir);
    //global rotation
    rayDir = float3x3::Rx(scene.cam.camRot.x)
                *(float3x3::Ry(scene.cam.camRot.y)
                    *(float3x3::Rz(scene.cam.camRot.z)
                        *rayDir));

    Ray r(scene.cam.camPos, rayDir);
    for(int i = 0; i < config.marchConfig.marchSteps; i++){
        //TODO: This might destroy performance due to warp desync and bad branch prediction. Verify this does not happen.
        //TODO: the selection of 128 is arbitrary here. It should probably be optimized or configurable
        if (i % 128 == 0 && ct.cancelled) {
            return debug_purple;
        }
        auto& bodies = scene.bodies;
        r.step(config.marchConfig.marchStepDeltaTime, bodies, scene.constants);
        for(int b = 0; b<bodies.elems; b++){
            if(length(bodies[b].position - r.position) <= bodies[b].radius){
                const auto& collided = bodies[b];

                const auto uv = uvMapSphere(
                    r.position,
                    collided.position,
                    collided.rotation
                    );
                auto color = collided.material.sample(uv);
                return color;
            }
        }
    }

    // if the ray hasn't hit anything after all steps, it is assumed to have hit skybox
    // this is intentionally passing direction and not position.
    // This effectively projects the ray where it would have gone in the skybox if it was stepped to infinity.
    // this prevents bugs where if the ray hasn't actually crossed the origin by the end of the render,
    // the ray would get projected onto the (for example) -X of the skybox despite it pointing towards +X.
    return scene.nohit.sample(uvMapSphere(
        r.direction,
        make_float3(0,0,0),
        make_float3(0,0,0))
        );
}

__device__ __host__ float2 calculate_screenspace_coords(const uint2 resolution, const uint2 index) {
    assert(index.x >= 0 && index.x < resolution.x);
    assert(index.y >= 0 && index.y < resolution.y);
    float2 idxFloat = make_float2(index.x, index.y);

    //screenspace xy coordinates in range [0,1]
    const float2 screenSpace = (idxFloat/make_float2(resolution));
    return screenSpace;
}

__global__ void renderFrameKernel(
    const Scene* const scene,
    const RenderConfig* const config,
    const CancellationToken* const ct,
    FrameBuffer* frameBuffer
    ){
    const unsigned int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    uint2 idx = make_uint2(xIdx, yIdx);
    if(xIdx >= config->resolution.x || yIdx >= config->resolution.y){
        return;
    }
    auto scr = calculate_screenspace_coords(config->resolution, idx);
    auto pixelColor = renderPixel(*scene, *config, *ct, scr);
    (*frameBuffer)[idx] = pixelColor;
}

std::optional<FrameBuffer> Renderer::renderGPU(const Scene& scene, const RenderConfig& config, const CudaContext& cudaContext, const CancellationToken &ct) {
    FrameBuffer buffer(config.resolution);
    scene.toDevice();
    config.toDevice();
    ct.toDevice();
    buffer.toDevice();

    if (ct.cancelled) {
        return {};
    }

    //max of 29 right now
    //24 seems optimal since we can't reach 32 due to register pressure
    unsigned int tdim = 22; //FIXED: Square tdim in the below equation
    int numBlocks = ((float)(config.resolution.x*config.resolution.y))/((float)(tdim*tdim));
    unsigned int bdim = (int)ceil(sqrt(numBlocks));

    // TODO: put this on the cuContext
    wrap_cuda([&]{return cudaDeviceSynchronize();});

    if (ct.cancelled) {
        return {};
    }

    // this must be launched on a stream in order for cancellation to work properly.
    renderFrameKernel<<<
        {bdim,bdim},
        {tdim,tdim},
        0,
        cudaContext.renderStream
    >>>(
        scene.device_ptr,
        config.device_ptr,
        ct.device_ptr,
        buffer.device_ptr
        );
    wrap_cuda([&]{return cudaStreamSynchronize(cudaContext.renderStream);});
    wrap_cuda([]{return cudaGetLastError();});

    if (ct.cancelled) {
        return {};
    }
    buffer.fromDevice();
    return buffer;
}

std::optional<FrameBuffer> Renderer::renderCPU(const Scene& scene, const RenderConfig& config, const CancellationToken &ct) {
    int numThreads = 12;

    //spawn threads
    std::vector<int> threadArgs(numThreads);
    int counter = 0;
    std::ranges::generate(threadArgs, [&counter] {
        return counter++;
    });

    auto frameBuffer = FrameBuffer(config.resolution);
    std::for_each(
        std::execution::par,
        threadArgs.begin(),
        threadArgs.end(),
        [&](const int tid) {
            for(int y = tid; y < config.resolution.y; y+=numThreads){
                for(int x = 0; x < config.resolution.x; x++){
                    const auto idx = make_uint2(x,y);
                    const auto scr = calculate_screenspace_coords(config.resolution, idx);
                    frameBuffer[idx] = renderPixel(scene, config, ct, scr);
                }
            }
        }
    );

    if (ct.cancelled) {
        return {};
    }

    return frameBuffer;
}

FrameBuffer Renderer::renderGPU(const Scene &scene, const RenderConfig &config, const CudaContext &cudaContext) {
    const auto ct = CancellationToken();
    return *renderGPU(scene, config, cudaContext, ct);
}
FrameBuffer Renderer::renderCPU(const Scene &scene, const RenderConfig &config) {
    const auto ct = CancellationToken();
    return *renderCPU(scene, config, ct);
}