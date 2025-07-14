#ifndef __RENDERING_H__
#define __RENDERING_H__

#include <vector_types.h>

#include "body.cuh"

#include "body.cuh"
#include "camera.cuh"
void renderGPU(int numFrames, camera c, 
                body bodies[], int bodiesCount,
                uint2 sunTextureDim, uchar3* sunTexture,
                uint2 starsTextureDim, uchar3* starsTexture);

void renderCPU(int numFrames, camera c, 
                body bodies[], int bodiesCount,
                uint2 sunTextureDim, uchar3* sunTexture,
                uint2 starsTextureDim, uchar3* starsTexture);
#endif