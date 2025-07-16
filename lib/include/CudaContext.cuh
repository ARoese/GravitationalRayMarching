//
// Created by atomr on 7/14/2025.
//

#ifndef CONTEXT_CUH
#define CONTEXT_CUH

class CudaContext {
    public:
    cudaStream_t renderStream;
    cudaStream_t copyStream;

    CudaContext() {
        cudaStreamCreate(&renderStream);
        cudaStreamCreate(&copyStream);
    }

    ~CudaContext() {
        cudaStreamDestroy(renderStream);
        cudaStreamDestroy(copyStream);
    }
};

#endif //CONTEXT_CUH
