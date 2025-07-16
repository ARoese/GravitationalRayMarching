//
// Created by atomr on 7/14/2025.
//

#include "Texture.cuh"

#include <assert.h>
#include <format>
#include <stdexcept>

#include "loadImage.hpp"
#include "Utils.cuh"

uchar3 Texture::sample(float2 uv) const {
    uint2 coords = {
        (unsigned int)(uv.x*(size.x-1)),
        (unsigned int)(uv.y*(size.y-1))
    };

    return operator[](coords);
}

Texture Texture::loadFromFile(const char* filename) {
    //load in sky-sphere texture
    uchar3* data;
    const uint2 starsDim = loadImage(filename, &data);
    return {starsDim, data};
}

Texture::Texture(uint2 size): size(size) {
    auto flatSize = size.x * size.y;
    data = new uchar3[flatSize];
    for (int i = 0; i < flatSize; i++) {
        data[i] = i % 2 == 0 ? uchar3{255, 0, 255} : uchar3{0, 0, 0};
    }
}

Texture::Texture(uint2 size, uchar3* data): data(data), size(size) {}

Texture::~Texture() {
    if (data != nullptr) {
        delete[] data;
    }
    if (device_data != nullptr) {
        wrap_cuda([&]{return cudaFree(device_data);});
    }
}

uchar3& Texture::operator[](uint2 index) {
    assert(index.x < size.x);
    assert(index.y < size.y);

    auto dataLoc = location == DEVICE ? device_data : data;
    return dataLoc[index.y*size.x + index.x];
}

const uchar3& Texture::operator[](uint2 index) const{
    assert(index.x < size.x);
    assert(index.y < size.y);

    auto dataLoc = location == DEVICE ? device_data : data;
    return dataLoc[index.y*size.x + index.x];
}

void Texture::toDeviceImpl(Texture* deviceLocation) const {
    if (device_data == nullptr) {
        cudaMalloc(const_cast<uchar3**>(&device_data), size.x*size.y*sizeof(uchar3));
        //wrap_cuda([&]{return  });
    }
    wrap_cuda([&]{return cudaMemcpy(device_data, data, size.x*size.y*sizeof(uchar3), cudaMemcpyHostToDevice);});
    simpleCopyToDevice(deviceLocation);
}


void Texture::fromDeviceImpl(Texture* deviceLocation) {
    if (device_data == nullptr) {
        throw std::invalid_argument("No gpu resource associated. (no toDevice call)");
    }
    wrap_cuda([&]{return cudaMemcpy(data, device_data, size.x*size.y*sizeof(uchar3), cudaMemcpyDeviceToHost);});
    simpleCopyFromDevice(deviceLocation);
}
