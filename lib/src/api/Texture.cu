//
// Created by atomr on 7/19/2025.
//

#include "Texture.cuh"

#include "compat_impl.cuh"
#include "loadImage.hpp"
#include "api/C_Texture.h"
#include "api/compat_wrappers.h"

extern "C" {
    COMPAT_DESTRUCTOR_IMPL(Texture)

    void save_to_file( char* const outFile, const COMPAT_PTRTYPE(Texture) texture) {
        const Texture* const tp = static_cast<Texture*>(texture);
        saveImage(
            outFile,
            tp->data,
            tp->size,
            0
            );
    }
}
