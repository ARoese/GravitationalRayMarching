//
// Created by atomr on 7/14/2025.
//

#include "Buffer.cuh"

#include <assert.h>
#include <format>

#include "Transferable.cuh"
#include "Utils.cuh"
#include <stdexcept>

#include "Body.cuh"
template class Buffer<Body>;

template<typename T>
Buffer<T>::Buffer(T* items, int elems): elems(elems), data(items) {}

template<typename T>
Buffer<T>::~Buffer() {
    delete[] data;
    wrap_cuda([&] {return cudaFree(device_data);});
}

template<typename T>
T &Buffer<T>::operator[](int index) {
    assert(index < elems && index >= 0);

    return data[index];

}

template<typename T>
const T &Buffer<T>::operator[](int index) const {
    assert(index < elems && index >= 0);

    if (this->location == Transferable<Buffer<T>>::Location::DEVICE) {
        return device_data[index];
    }

    return data[index];
}

template<typename T>
void Buffer<T>::toDeviceImpl(Buffer<T>* deviceLocation) const {
    if (device_data == nullptr) {
        wrap_cuda([&]{return cudaMalloc(const_cast<T**>(&device_data), elems*sizeof(T));});
    }
    wrap_cuda([&]{return cudaMemcpy(device_data, data, elems*sizeof(T), cudaMemcpyHostToDevice);});
    this->simpleCopyToDevice(deviceLocation);
    for (int i = 0; i < elems; i++) {
        operator[](i).toDevice(&device_data[i]);
    }
}


template<typename T>
void Buffer<T>::fromDeviceImpl(Buffer<T>* deviceLocation) {
    if (device_data == nullptr) {
        throw std::invalid_argument("No gpu resource associated. (no toDevice call)");
    }
    wrap_cuda([&]{return cudaMemcpy(data, device_data, elems*sizeof(T), cudaMemcpyDeviceToHost);});
    Transferable<Buffer<T>>::simpleCopyFromDevice(deviceLocation);

    for (int i = 0; i < elems; i++) {
        operator[](i).fromDevice(&device_data[i]);
    }
}