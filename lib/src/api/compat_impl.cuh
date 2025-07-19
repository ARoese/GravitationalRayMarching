//
// Created by atomr on 7/18/2025.
//

#ifndef COMPAT_IMPL_CUH
#define COMPAT_IMPL_CUH

#define COMPAT_DESTRUCTOR_IMPL(COMPAT_CLASS_NAME) \
    COMPAT_DESTRUCTOR(COMPAT_CLASS_NAME) {                                      \
        auto toDestroy_cast = static_cast<##COMPAT_CLASS_NAME## *>(toDestroy);  \
        delete toDestroy_cast;                                                  \
    }

#endif //COMPAT_IMPL_CUH
