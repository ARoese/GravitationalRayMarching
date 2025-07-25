//
// Created by atomr on 7/14/2025.
//

#ifndef TRANSFERABLE_CUH
#define TRANSFERABLE_CUH
#include <stdexcept>
#include <cuda_runtime_api.h>
#include "assert.h"
#include "Utils.cuh"
#include <mutex>

/**
 * Object whose members can be copied to and from the device.
 */
template<typename BASE>
class Transferable {
public:
    enum Location {
        HOST,
        DEVICE
    } location;

    mutable std::recursive_mutex transferable_mtx;

    BASE* device_ptr = nullptr;

    Transferable(): location(HOST), device_ptr(nullptr) {}
    Transferable(Transferable&& o) noexcept :
        location(o.location) {
        std::lock_guard<std::recursive_mutex> lock(o.transferable_mtx);
        std::exchange(o.device_ptr, nullptr);
        // free owned resources, because they'll not be compatible with the new context
        if (device_ptr != nullptr) {
            wrap_cuda([&]{return cudaFree(device_ptr);});
            device_ptr = nullptr;
        }
    }
    Transferable& operator=(Transferable&& o) = default;
    ~Transferable() {
        freeGpuResources();
    }

    // copy this object's contents to the device
    __host__ void toDevice(BASE* deviceLocation) const {
        std::lock_guard<std::recursive_mutex> lock(transferable_mtx);
        if (deviceLocation != nullptr) { // we're owned by someone else
            if (device_ptr != nullptr) {
                // we should no longer own resources on the gpu
                const_cast<Transferable*>(this)->freeGpuResources();
            }
            static_cast<const BASE*>(this)->toDeviceImpl(deviceLocation);
        }else {
            // if we're unowned but own no memory, then make it
            if (device_ptr == nullptr) {
                wrap_cuda([&]{return cudaMalloc(const_cast<BASE**>(&device_ptr), sizeof(BASE));});
            }
            static_cast<const BASE*>(this)->toDeviceImpl(device_ptr);
        }
    }

    __host__ void toDevice() const {
        toDevice(nullptr);
    }

    __host__ void fromDevice() {
        fromDevice(nullptr);
    }

    // copy this object's contents from the device
    __host__ void fromDevice(BASE* deviceLocation) {
        std::lock_guard<std::recursive_mutex> lock(transferable_mtx);
        if (deviceLocation != nullptr) {
            if (device_ptr != nullptr) {
                // we should no longer own resources on the gpu
                freeGpuResources();
            }
            static_cast<BASE*>(this)->fromDeviceImpl(deviceLocation);
        } else {
            if (device_ptr == nullptr) {
                throw std::invalid_argument("No associated device resource (no toDevice call)");
            }
            static_cast<BASE*>(this)->fromDeviceImpl(device_ptr);
        }
    }
protected:
    void freeGpuResources() {
        std::lock_guard<std::recursive_mutex> lock(transferable_mtx);
        if (device_ptr != nullptr) {
            wrap_cuda([&]{return cudaFree(device_ptr);});
            device_ptr = nullptr;
        }
    }

    __host__ void simpleCopyFromDevice(BASE* deviceLocation) {
        wrap_cuda([&]{return cudaMemcpy(this, deviceLocation, sizeof(BASE), cudaMemcpyDeviceToHost);});
        location = HOST;
    }

    __host__ void simpleCopyToDevice(BASE* deviceLocation) const {
        *const_cast<Location*>(&location) = DEVICE;
        wrap_cuda([&]{return cudaMemcpy(deviceLocation, this, sizeof(BASE), cudaMemcpyHostToDevice);});
        *const_cast<Location*>(&location) = HOST;
    }
    __host__ void toDeviceImpl(BASE* deviceLocation) const {
        simpleCopyToDevice(deviceLocation);
    }
    __host__ void fromDeviceImpl(BASE* deviceLocation) {
        simpleCopyFromDevice(deviceLocation);
    }
};

#endif //TRANSFERABLE_CUH
