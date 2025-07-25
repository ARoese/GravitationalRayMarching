//
// Created by atomr on 7/14/2025.
//

#ifndef UTILS_CUH
#define UTILS_CUH
#include <cuda_runtime_api.h>
#include <functional>

void wrap_cuda(std::function<cudaError_t()> action);

#endif //UTILS_CUH
