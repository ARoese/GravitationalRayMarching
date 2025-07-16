//
// Created by atomr on 7/14/2025.
//

#ifndef TEXTURE_CUH
#define TEXTURE_CUH
#include "Buffer.cuh"
#include "Transferable.cuh"

class Texture: public Transferable<Texture> {
public:
    uchar3* data;
    uchar3* device_data = nullptr;
    uint2 size;

    __host__ Texture(uint2 size, uchar3* data);
    __host__ Texture(uint2 size);
    Texture( Texture&& o ) noexcept :
        data( std::exchange( o.data, nullptr )),
        device_data(std::exchange( o.device_data, nullptr)),
        size(o.size){}

    __host__ ~Texture();
    __host__ __device__ uchar3& operator[](uint2 index);
    __host__ __device__ const uchar3& operator[](uint2 index) const;
    __host__ __device__ uchar3 sample(float2 uv) const;
    static __host__ Texture loadFromFile(const char* filename);

    __host__ void toDeviceImpl(Texture* deviceLocation) const;
    __host__ void fromDeviceImpl(Texture* deviceLocation);
};

typedef Texture FrameBuffer;

#endif //TEXTURE_CUH
