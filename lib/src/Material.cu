//
// Created by atomr on 7/14/2025.
//

#include "Material.cuh"
#include <assert.h>
void Material::toDeviceImpl(Material *deviceLocation) const {
    simpleCopyToDevice(deviceLocation);
    switch (type) {
        case TEXTURE:
            texture.toDevice(&deviceLocation->texture);
            break;
        case FLAT:
            break;
    }
}

void Material::fromDeviceImpl(Material *deviceLocation) {
    simpleCopyFromDevice(deviceLocation);
    switch (type) {
        case TEXTURE:
            texture.fromDevice(&deviceLocation->texture);
            break;
        case FLAT:
            break;
    }
}

Material::~Material() {
    switch (type) {
        case TEXTURE:
            texture.~Texture();
            break;
        case FLAT:
            break;
    }
}

uchar3 Material::sample(float2 uv) const {
    switch (type) {
        case TEXTURE:
            return texture.sample(uv);
        case FLAT:
            return color;
    }
    assert(false);
}