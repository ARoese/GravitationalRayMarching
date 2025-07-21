//
// Created by atomr on 7/20/2025.
//

#ifndef CANCELLATIONTOKEN_CUH
#define CANCELLATIONTOKEN_CUH

#include "Transferable.cuh"

/**
 * Thread-safe indicator for cancellation of an operation. Call cancel() and any operations
 * associated with this token will cancel at their earliest convenience. The result of canceled
 * operations is operation-defined.
 */
struct CancellationToken : Transferable<CancellationToken> {
    volatile bool cancelled = false;

    void cancel();
    bool isCancelled();
    __host__ void toDeviceImpl(CancellationToken* deviceLocation) const;
};

#endif //CANCELLATIONTOKEN_CUH
