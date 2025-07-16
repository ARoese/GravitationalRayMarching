//
// Created by atomr on 7/14/2025.
//

#ifndef SCENE_CUH
#define SCENE_CUH

#include "body.cuh"
#include "camera.cuh"
#include <cuda/std/array>
class Scene: public Transferable<Scene> {
    public:
    Camera cam;
    Buffer<Body> bodies;
    UniversalConstants constants;
    Material nohit;

    Scene(Camera cam, Buffer<Body> bodies, Material nohit, UniversalConstants constants) :
        cam(cam),
        bodies(std::move(bodies)),
        nohit(std::move(nohit)),
        constants(constants){}

    Scene(Camera cam, Buffer<Body> bodies, Material nohit) :
        cam(cam),
        bodies(std::move(bodies)),
        nohit(std::move(nohit)),
        constants(real_universal_constants()){}

    Scene(Scene&& o) noexcept:
        cam(o.cam),
        bodies(std::move(o.bodies)),
        nohit(std::move(o.nohit)) {}

    __host__ void toDeviceImpl(Scene* deviceLocation) const;
    __host__ void fromDeviceImpl(Scene* deviceLocation);
};

#endif //SCENE_CUH
