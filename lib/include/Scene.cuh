//
// Created by atomr on 7/14/2025.
//

#ifndef SCENE_CUH
#define SCENE_CUH

#include <memory>

#include "body.cuh"
#include "camera.cuh"
#include <cuda/std/array>
#include <cuda_runtime_api.h>
#include "api/D_Scene.h"

class Scene: public Transferable<Scene> {
    public:
    Camera cam;
    Buffer<Body> bodies;
    UniversalConstants constants;
    Material nohit;

    Scene(const Camera& cam, Buffer<Body> bodies, Material nohit, const UniversalConstants& constants) :
        cam(cam),
        bodies(std::move(bodies)),
        constants(constants),
        nohit(std::move(nohit)){}

    Scene(const Camera& cam, Buffer<Body> bodies, Material nohit) :
        cam(cam),
        bodies(std::move(bodies)),
        constants(real_universal_constants),
        nohit(std::move(nohit)){}

    Scene(const D_Scene& o) : cam(o.cam), bodies(Buffer<Body>(nullptr, 0)), constants(o.constants), nohit(o.nohit) {
        auto bodies_arr = new Body[o.n_bodies];
        for (int i = 0; i < o.n_bodies; i++) {
            new(&bodies_arr[i]) Body(o.bodies[i]);
        }

        new(&bodies) Buffer(bodies_arr, o.n_bodies);
    }

    Scene(Scene&& o) noexcept:
        cam(o.cam),
        bodies(std::move(o.bodies)),
        constants(o.constants),
        nohit(std::move(o.nohit)) {}

    __host__ void toDeviceImpl(Scene* deviceLocation) const;
    __host__ void fromDeviceImpl(Scene* deviceLocation);
};

#endif //SCENE_CUH
