//
// Created by atomr on 7/14/2025.
//

#ifndef BUFFER_CUH
#define BUFFER_CUH
#include "Transferable.cuh"

template<typename T>
class Buffer : public Transferable<Buffer<T>>{
public:
    unsigned int elems;
    T* data;
    T* device_data = nullptr;
    __host__ Buffer(T* items, int elems);
    __host__ Buffer(Buffer&& o) noexcept:
        elems(o.elems),
        data(std::exchange(o.data, nullptr)),
        device_data(std::exchange(o.device_data, nullptr)) {};
    __host__ ~Buffer();
    __host__ __device__ T& operator[](int index);
    __host__ __device__ const T& operator[](int index) const;
    __host__ void toDeviceImpl(Buffer<T>* deviceLocation) const;
    __host__ void fromDeviceImpl(Buffer<T>* deviceLocation);
};

#endif //BUFFER_CUH
