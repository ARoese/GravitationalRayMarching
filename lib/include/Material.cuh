//
// Created by atomr on 7/14/2025.
//

#ifndef MATERIAL_CUH
#define MATERIAL_CUH
#include "api/D_Material.h"
#include "Texture.cuh"
#include "Transferable.cuh"

class Material: public Transferable<Material> {
public:
    enum Type {
        TEXTURE,
        FLAT
    } type;

    Type fromDomain(D_Material::Type o) {
        switch (o) {
            case D_Material::Type::TEXTURE:
                return TEXTURE;
            case D_Material::Type::FLAT:
                return FLAT;
        }
        throw std::invalid_argument("Invalid Material Type");
    }

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

    __host__ Material(D_Material o): type(fromDomain(o.type)) {
        if (type == FLAT) {
            color = {o.color.x, o.color.y, o.color.z,};
        }else {
            new(&texture) Texture(Texture::loadFromFile(o.texture_path));
        }
    }

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
