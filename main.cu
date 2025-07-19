#include <stdio.h>
#include <iostream>

//https://github.com/kashif/cuda-workshop/blob/master/cutil/inc/cutil_math.h
#include <filesystem>

#include "camera.cuh"
#include "cutil_math.cuh"
#include "loadImage.hpp"
#include "Renderer.cuh"
#include "Scene.cuh"
#include "Texture.cuh"
#include "lib/include/api/UniversalConstants.h"

void printUsage(char* programName){
    printf("USAGE: %s device imageDim num_frames debug\n", programName);
    printf("device: cpu or gpu. Defaults to gpu unless cpu is selected.\n");
    printf("imageDim: Dimensions of output image.\n");
    printf("num_frames: How many frames to render.\n");
    printf("debug: true/false, whether to use debug uv textures\n");
}

Body makeBody( float radius, float mass, float3 position, float3 rotation, Material mat, UniversalConstants constants) {
    auto schwarz = Body::getSchwarzschildRadiusForMass(mass, constants);
    return Body(
        max(radius, schwarz),
        mass,
        position,
        rotation,
        radius < schwarz ? Material({0,0,0}) : std::move(mat));
}

void renderRotationCenteredOn(
    std::string destFolder,
    int numFrames,
    Scene scene,
    Body& center,
    float distanceFromCenter,
    std::function<FrameBuffer(Scene&)> renderScene
    ) {
    float3 camRotCenter = center.position; //point that the camera rotates around
    for (int i = 0; i < numFrames; i++) {
        std::cout << std::format("Rendering frame #{}", i) << std::endl;
        unsigned int camDistance = 250; //distance of camera from center object

        //update camera position
        float angle = ((float)i/numFrames)*2*M_PI;
        scene.cam.camPos = {cos(angle),sin(angle),0};
        scene.cam.camPos *= camDistance;
        scene.cam.camPos += camRotCenter;
        scene.cam.camRot = {0,0,angle+(float)M_PI};

        //render image
        FrameBuffer frameResult = renderScene(scene);
        auto destination = std::filesystem::path(destFolder)
            .append("out.png")
            .string();
        saveImage(destination.c_str(), frameResult.data, frameResult.size, i);
    }
}

//3840,2160 is 4K
int main(int argc, char* argv[]) {
    // take care of command line stuff
    unsigned int numFrames = 16;
    uint imageDim = 1024;
    //const char* starsPath = "assets/test_uv.jpg";
    const char* starsPath = "assets/8k_stars_milky_way.jpg";
    bool renderOnCPU = false;
    if(argc == 1){
        printf("NOTICE: Using defaults\n");
    }else if(argc == 5){
        renderOnCPU = (strcmp(argv[1], "cpu") == 0) || (strcmp(argv[1], "CPU") == 0);
        imageDim = atoi(argv[2]);
        numFrames = atoi(argv[3]); //numFrames
        if(strcmp(argv[4], "true") == 0){ //use debug uv
            starsPath = "assets/test_uv.jpg";
        }
    }else{
        printUsage(argv[0]);
        return 1;
    }
    
    if(numFrames <= 0){
        printf("ERROR: argument num_frames ('%s') is not valid.\n", argv[3]);
        printUsage(argv[0]);
        return 1;
    }else if(imageDim <= 0){
        printf("ERROR: argument image_dims ('%s') is not valid.\n", argv[2]);
        printUsage(argv[0]);
        return 1;
    }

    printf("rendering on %s\n", renderOnCPU ? "CPU" : "GPU");

    //UniversalConstants constants = real_universal_constants();
    UniversalConstants constants = { .G = 6.6743e-11, .C = 10 };
    
    // set up scene
    auto bodies = new Body[1] {
        //makeBody(20, 0, {250,-60,0},{0,0,0}, Material(Texture::loadFromFile("assets/8k_sun.jpg")), constants),
        makeBody(6, 1e11, {250,0,0},{0,0,0}, Material({0,0,0}), constants)
        //body(6, 0, {140,4,0},{0,0,0},{0,128,0}),
    };

    auto scene = Scene(
        Camera(
                {50*DEG2RAD, 50*DEG2RAD},
                {0,0,0},
                {0,0,0},
                {imageDim,imageDim}
            ),
            Buffer<Body>(bodies, 1),
            Material(Texture::loadFromFile(starsPath)),
            constants
    );

    Renderer renderer;
    CudaContext cuContext;

    auto renderScene = [&](Scene& scene) {
        return renderOnCPU ? renderer.renderCPU(scene) : renderer.renderGPU(scene, cuContext);
    };

    renderRotationCenteredOn(
        "outputs/",
        numFrames, std::move(scene), scene.bodies[0],
        250, renderScene);


    printf("Done rendering\n");
    return 0;
}