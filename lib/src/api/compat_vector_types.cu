//
// Created by atomr on 7/19/2025.
//
#include "compat_vector_types.cuh"
__host__ __device__ uchar3 fromCompat(const d_uchar3& o) {
    return uchar3(o.x, o.y, o.z);
}

__host__ __device__ uint2 fromCompat( const d_uint2& o) {
    return uint2(o.x, o.y);
}

__host__ __device__ float3 fromCompat(const d_float3& o) {
    return float3(o.x, o.y, o.z);
}

__host__ __device__ float2 fromCompat(const d_float2& o) {
    return float2(o.x, o.y);
}