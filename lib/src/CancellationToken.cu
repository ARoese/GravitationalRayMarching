//
// Created by atomr on 7/20/2025.
//

#include "CancellationToken.cuh"

void CancellationToken::cancel() {
    std::lock_guard<std::recursive_mutex> lock(transferable_mtx);
    cancelled = true;
    if (device_ptr != nullptr) {
        toDevice();
    }
}

__host__ void CancellationToken::toDeviceImpl(CancellationToken* deviceLocation) const {
    // this stream is necessary because dispatches on the null stream will not overlap. We want this to overlap.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    wrap_cuda([&] {
        return cudaMemcpyAsync(deviceLocation, this, sizeof(CancellationToken), cudaMemcpyHostToDevice, stream);
    });

    wrap_cuda([&] {return cudaStreamSynchronize(stream);});
}