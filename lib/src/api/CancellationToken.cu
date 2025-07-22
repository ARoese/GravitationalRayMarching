//
// Created by atomr on 7/21/2025.
//

#include "CancellationToken.cuh"

#include "api/C_CancellationToken.h"
#include "compat_impl.cuh"

extern "C" {
    COMPAT_PTRTYPE(CancellationToken) make_CancellationToken() {
        return new CancellationToken();
    }
    COMPAT_DESTRUCTOR_IMPL(CancellationToken);

    void ct_cancel(COMPAT_PTRTYPE(CancellationToken) ct) {
        const auto ct_cast = static_cast<CancellationToken *>(ct);
        ct_cast->cancel();
    }
    bool ct_wasCancelled(COMPAT_PTRTYPE(CancellationToken) ct) {
        const auto ct_cast = static_cast<CancellationToken *>(ct);
        return ct_cast->cancelled;
    }
}
