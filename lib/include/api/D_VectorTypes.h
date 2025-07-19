//
// Created by atomr on 7/17/2025.
//

#ifndef D_VECTORTYPES_H
#define D_VECTORTYPES_H

#ifdef __cplusplus
    extern "C" {
#endif

        typedef struct {
            unsigned char x;
            unsigned char y;
            unsigned char z;
        } d_uchar3;

        typedef struct {
            unsigned int x;
            unsigned int y;
        } d_uint2;

        typedef struct {
            float x;
            float y;
            float z;
        } d_float3;

        typedef struct {
            float x;
            float y;
        } d_float2;

#ifdef __cplusplus
    }
#endif

#endif //D_VECTORTYPES_H
