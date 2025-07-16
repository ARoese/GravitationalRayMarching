//
// Created by atomr on 7/14/2025.
//

#include "Scene.cuh"
void Scene::toDeviceImpl(Scene* deviceLocation) const {
    simpleCopyToDevice(deviceLocation);
    bodies.toDevice(&deviceLocation->bodies);
    nohit.toDevice(&deviceLocation->nohit);
}

void Scene::fromDeviceImpl(Scene* deviceLocation) {
    simpleCopyFromDevice(deviceLocation);
    bodies.fromDevice(&deviceLocation->bodies);
    nohit.fromDevice(&deviceLocation->nohit);
}

