//
// Created by atomr on 7/19/2025.
//

#ifndef C_TEXTURE_H
#define C_TEXTURE_H
#include "compat_wrappers.h"

#ifdef __cplusplus
    extern "C" {
#endif

        typedef void* COMPAT_PTRTYPE(Texture);
        //C_Texture_Ptr make_Texture(int x, int y);

        void destroy_Texture(C_Texture_Ptr toDestroy);

        void save_to_file( char* outFile, COMPAT_PTRTYPE(Texture) texture );

#ifdef __cplusplus
    }
#endif

#endif //C_TEXTURE_H
