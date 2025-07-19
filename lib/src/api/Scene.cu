//
// Created by atomr on 7/18/2025.
//

#include "Scene.cuh"

#include "compat_impl.cuh"
#include "compat_vector_types.cuh"
#include "api/compat_wrappers.h"
#include "api/D_Scene.h"

extern "C" {
    COMPAT_CONSTRUCTOR(Scene) {
        return new Scene(dc);
    }

    COMPAT_DESTRUCTOR_IMPL(Scene)
}
