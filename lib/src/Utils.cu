//
// Created by atomr on 7/14/2025.
//

#include "Utils.cuh"
void wrap_cuda(std::function<cudaError_t()> action) {
    auto res = action();
    if (res != cudaSuccess) {
        throw res;
    }
}
