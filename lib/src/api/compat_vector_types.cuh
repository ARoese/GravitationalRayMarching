//
// Created by atomr on 7/18/2025.
//

#ifndef COMPAT_VECTOR_TYPES_CUH
#define COMPAT_VECTOR_TYPES_CUH
#include "api/D_VectorTypes.h"

__host__ __device__ uchar3 fromCompat(const d_uchar3& o);
__host__ __device__ uint2 fromCompat( const d_uint2& o);
__host__ __device__ float3 fromCompat(const d_float3& o);
__host__ __device__ float2 fromCompat(const d_float2& o);

#endif //COMPAT_VECTOR_TYPES_CUH
