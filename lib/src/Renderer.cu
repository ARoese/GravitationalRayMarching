#include "Renderer.cuh"

#include <algorithm>
#include <execution>
#include <format>
#include <iostream>
#include <stdio.h>
#include <vector>

#include "cutil_math.cuh"
#include "cmatrix.cuh"

#include "universalConstants.hpp"
#include "Ray.cuh"
#include "loadImage.hpp"

__device__ __host__ float2 uvMapSphere(const float3 rayPos, const float3 bodyPos, const float3 bodyRot){
    //https://en.wikipedia.org/wiki/UV_mapping
    float3 d = normalize(rayPos - bodyPos);

    d = float3x3::Rx(bodyRot.x)*(float3x3::Ry(bodyRot.y)*(float3x3::Rz(bodyRot.z)*d));

    float u = 0.5 + atan2(d.z, d.x)/(2*M_PI);
    float v = 0.5 + asin(d.y)/M_PI;

    return {u,v};
}

__device__ __host__ uchar3 renderPixel(const Scene& scene, uint2 idx){
    float2 idxFloat = make_float2(idx.x, idx.y);
    
    //screenspace xy coordinates in range [-0.5,0.5]
    const float2 screenSpace = (idxFloat/make_float2(scene.cam.resolution)) - 0.5;
    //angular screenspace coordinates from [-fov.x, fov.x] and so on
    const float2 screenAngle = screenSpace * scene.cam.fov;

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
    #pragma unroll
    for(int i = 0; i < 10000; i++){
        auto& bodies = scene.bodies;
        r.step(0.005, bodies, scene.constants);
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

    //if the ray hasn't hit anything after all steps, it is assumed to have hit skybox
    return scene.nohit.sample(uvMapSphere(
        r.position,
        make_float3(0,0,0),
        make_float3(0,0,0))
        );
}

__global__ void renderFrameKernel(Scene* scene, FrameBuffer* frameBuffer){
    int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    uint2 idx = make_uint2(xIdx, yIdx);
    if(xIdx >= scene->cam.resolution.x || yIdx >= scene->cam.resolution.y){
        return;
    }

    auto pixelColor = renderPixel(*scene, idx);
    (*frameBuffer)[idx] = pixelColor;
}

FrameBuffer Renderer::renderGPU(Scene& scene, CudaContext& cudaContext) {
    FrameBuffer buffer(scene.cam.resolution);
    scene.toDevice();
    buffer.toDevice();

    //max of 29 right now
    //24 seems optimal since we can't reach 32 due to register pressure
    unsigned int tdim = 22; //FIXED: Square tdim in the below equation
    int numBlocks = ((float)(scene.cam.resolution.x*scene.cam.resolution.y))/((float)(tdim*tdim));
    unsigned int bdim = (int)ceil(sqrt(numBlocks));

    // TODO: put this on the cuContext
    wrap_cuda([&]{return cudaDeviceSynchronize();});

    renderFrameKernel<<<{bdim,bdim},{tdim,tdim}, 0, cudaContext.renderStream>>>(scene.device_ptr, buffer.device_ptr);
    wrap_cuda([&]{return cudaStreamSynchronize(cudaContext.renderStream);});
    wrap_cuda([]{return cudaGetLastError();});

    buffer.fromDevice();
    return buffer;
}

/* void renderFramesGPU(Scene scene, CudaContext cudaContext){
    //max of 29 right now
    //24 seems optimal since we can't reach 32 due to register pressure
    unsigned int tdim = 24; //FIXED: Square tdim in the below equation
    int numBlocks = ((float)(c.resolution.x*c.resolution.y))/((float)(tdim*tdim));
    unsigned int bdim = (int)ceil(sqrt(numBlocks));

    //wait for it to finish
    cudaDeviceSynchronize();
    //check for errors
    std::cout << std::format("%s", cudaGetErrorString(cudaGetLastError())) << std::endl;

    std::cout
        << std::format("starting kernel with dims {},{} ({})", bdim,bdim,tdim)
        << std::endl;

    float3 camRotCenter = {250,0,0}; //point that the camera rotates around
    unsigned int camDistance = 250; //distance of camera from center object
    for(int i = 0; i < numFrames; i++){
        //update camera position
        float angle = ((float)i/numFrames)*2*M_PI;
        c.camPos = {cos(angle),sin(angle),0};
        c.camPos *= camDistance;
        c.camPos += camRotCenter;
        c.camRot = {0,0,angle+(float)M_PI};
        //send new camera orientation to device
        cudaMemcpyAsync(cameraDev, &c, sizeof(Camera), cudaMemcpyHostToDevice, copyStream);
        cudaStreamSynchronize(copyStream);
        std::cout
            << std::format("Kernel {} set up... Waiting on previous operations.", i)
            << std::endl;
        cudaStreamSynchronize(renderStream); //wait for the last kernel to finish

        //render image
        renderFrameKernel<<<{bdim,bdim},{tdim,tdim}, 0, renderStream>>>
            (cameraDev, bodiesDev, bodiesCount, frameBufferDev, starsDev, starsTextureDim);
        
        //while that is running, save the last frame asynchronously
        std::cout << std::format("Kernel {} launched...", i) << std::endl;
        // our save should not operate on the regions currently being worked on
        std::swap(cameraDev, cameraDev2);
        std::swap(frameBufferDev, frameBufferDev2);
        if(i != 0){ //if it's the first invocation, we don't have a previous frame to save
            //copy previous frame back from device and save it
            cudaMemcpyAsync(frameBuffer, frameBufferDev, frameBufferSize, cudaMemcpyDeviceToHost, copyStream);
            cudaStreamSynchronize(copyStream);
            saveImage(
                "outputs/out.png",
                frameBuffer, 
                {(uint)c.resolution.x, (uint)c.resolution.y}, i-1);
            std::cout << std::format("Saved result of kernel {}...", i-1) << std::endl;
        }
    }
    cudaStreamDestroy(copyStream);
    cudaStreamDestroy(renderStream);

    //save final frame
    cudaMemcpy(frameBuffer, frameBufferDev2, frameBufferSize, cudaMemcpyDeviceToHost);
    saveImage(
        "outputs/out.png",
        frameBuffer, 
        {(uint)c.resolution.x, (uint)c.resolution.y}, numFrames-1);
    std::cout << std::format("Saved result of kernel {}...", numFrames-1) << std::endl;
    
    //check for errors
    std::cout << std::format("{}", cudaGetErrorString(cudaGetLastError())) << std::endl;
    
    //free gpu resources
    cudaFree(bodiesDev);
    cudaFree(cameraDev);
    cudaFree(cameraDev2);
    cudaFree(frameBufferDev);
    cudaFree(frameBufferDev2);
    cudaFree(starsDev);
}
*/

FrameBuffer renderCPUFrame(Scene& scene){
    int numThreads = 12;

    //spawn threads
    std::vector<int> threadArgs(numThreads);
    int counter = 0;
    std::ranges::generate(threadArgs, [&counter] {
        return counter++;
    });

    FrameBuffer frameBuffer = FrameBuffer(scene.cam.resolution);
    std::for_each(
        std::execution::par,
        threadArgs.begin(),
        threadArgs.end(),
        [&](const int tid) {
            for(int y = tid; y < scene.cam.resolution.y; y+=numThreads){
                for(int x = 0; x < scene.cam.resolution.x; x++){
                    const auto idx = make_uint2(x,y);
                    frameBuffer[idx] = renderPixel(scene, idx);
                }
            }
        }
    );

    return frameBuffer;
}

FrameBuffer Renderer::renderCPU(Scene& scene) {
    return renderCPUFrame(scene);
}

void renderCPU(int numFrames, Scene scene){
    float3 camRotCenter = {250,0,0}; //point that the camera rotates around
    unsigned int camDistance = 250; //distance of camera from center object
    for(int i = 0; i < numFrames; i++){
        //update camera position
        float angle = ((float)i/numFrames)*2*M_PI;
        scene.cam.camPos = {cos(angle),sin(angle),0};
        scene.cam.camPos *= camDistance;
        scene.cam.camPos += camRotCenter;
        scene.cam.camRot = {0,0,angle+(float)M_PI};
        //send new camera orientation to device
        std::cout << std::format("Rendering frame {}...", i) << std::endl;

        //render image
        auto output = renderCPUFrame(scene);
        
        saveImage(
            "outputs/out.png",
            output.data,
            {output.size.x, output.size.y}, i);
        std::cout << std::format("Saved frame {}", i) << std::endl;
    }
        
}