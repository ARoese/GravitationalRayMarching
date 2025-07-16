//
// Created by atomr on 7/14/2025.
//

#ifndef MATERIAL_CUH
#define MATERIAL_CUH
#include "Texture.cuh"
#include "Transferable.cuh"

class Material: public Transferable<Material> {
public:
    enum Type {
        TEXTURE,
        FLAT
    } type;

    union {
        uchar3 color;
        Texture texture;
    };
    __host__ __device__ uchar3 sample(float2 uv) const;

    __host__ Material(uchar3 color): color(color) {
        type = FLAT;
    };
    __host__ Material(Texture tex): texture(std::move(tex)) {
        type = TEXTURE;
    };

    __host__ Material(Material&& o) noexcept: type(o.type) {
        if (type == TEXTURE) {
            new(&texture) Texture(std::move(o.texture));
        }if (type == FLAT) {
            color = o.color;
        }
    }
    __host__ void toDeviceImpl(Material* deviceLocation) const;
    __host__ void fromDeviceImpl(Material* deviceLocation);
    __host__ ~Material();
};

#endif //MATERIAL_CUH
