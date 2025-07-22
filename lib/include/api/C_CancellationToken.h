//
// Created by atomr on 7/21/2025.
//

#ifndef C_CANCELLATIONTOKEN_H
#define C_CANCELLATIONTOKEN_H
#include "compat_wrappers.h"
#ifdef __cplusplus
    extern "C" {
#endif

        typedef void *COMPAT_PTRTYPE(CancellationToken);
        COMPAT_PTRTYPE(CancellationToken) make_CancellationToken();
        COMPAT_DESTRUCTOR(CancellationToken);

        void ct_cancel(COMPAT_PTRTYPE(CancellationToken));
        bool ct_wasCancelled(COMPAT_PTRTYPE(CancellationToken));

#ifdef __cplusplus
        }
#endif

#endif //C_CANCELLATIONTOKEN_H
