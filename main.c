//
// Created by atomr on 7/19/2025.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "D_Body.h"
#include "D_Material.h"
#include "D_Scene.h"
#include "D_Camera.h"
#include "C_Texture.h"
#include "C_Rendering.h"
#include "D_RenderConfig.h"
#include "UniversalConstants.h"

void printUsage(char* programName){
    printf("USAGE: %s device imageDim num_frames debug\n", programName);
    printf("device: cpu or gpu. Defaults to gpu unless cpu is selected.\n");
    printf("imageDim: Dimensions of output image.\n");
    printf("num_frames: How many frames to render.\n");
    printf("debug: true/false, whether to use debug uv textures\n");
}

D_Body makeBody( float radius, float mass, d_float3 position, d_float3 rotation, D_Material mat, UniversalConstants constants) {
    float schwarz = getSchwarzschildRadiusForMass(mass, constants);
    D_Material black_material = {FLAT,{0, 0, 0}};
    D_Body body = {
        radius > schwarz ? radius : schwarz,
            mass,
            position,
            rotation,
            radius < schwarz ? black_material : mat
        };
    return body;
}

typedef char bool;
#define false 0;
#define true 1;

//3840,2160 is 4K
int main(int argc, char* argv[]) {
    // take care of command line stuff
    unsigned int imageDim = 1024;
    //const char* starsPath = "assets/test_uv.jpg";
    const char* starsPath = "assets/8k_stars_milky_way.jpg";
    bool renderOnCPU = false;
    if(argc == 1){
        printf("NOTICE: Using defaults\n");
    }else if(argc == 4){
        renderOnCPU = (strcmp(argv[1], "cpu") == 0) || (strcmp(argv[1], "CPU") == 0);
        imageDim = atoi(argv[2]);
        if(strcmp(argv[3], "true") == 0){ //use debug uv
            starsPath = "assets/test_uv.jpg";
        }
    }else{
        printUsage(argv[0]);
        return 1;
    }

    if(imageDim <= 0){
        printf("ERROR: argument image_dims ('%s') is not valid.\n", argv[2]);
        printUsage(argv[0]);
        return 1;
    }

    printf("rendering on %s\n", renderOnCPU ? "CPU" : "GPU");

    //UniversalConstants constants = real_universal_constants();
    UniversalConstants constants = { .G = 6.6743e-11, .C = 10 };

    d_float3 sunPos = {250,-60,0};
    D_Material sunMaterial = {TEXTURE, .texture_path="assets/8k_sun.jpg"};
    d_float3 zero = {0,0,0};
    d_float3 bhPos = {250,0,0};
    D_Material bhMaterial = {FLAT, {0,0,0}};
    // set up scene
    D_Body bodies[2] = {
        makeBody(20,0,sunPos,zero, sunMaterial, constants),
        makeBody(6, 1e11, bhPos,zero, bhMaterial, constants)
        //body(6, 0, {140,4,0},{0,0,0},{0,128,0}),
    };

    const D_Scene scene = {
        {
                {50*DEG2RAD, 50*DEG2RAD},
                {0,0,0},
                {0,0,0}
            },
            &bodies[0],
            2,
            constants,
            {TEXTURE, .texture_path=starsPath},

    };

    C_Scene_Ptr scene_ptr = make_Scene(scene);
    D_RenderConfig config = {
        {imageDim, imageDim},
        default_MarchConfig
    };
    const C_Texture_Ptr render_result = renderOnCPU
        ? render_cpu(scene_ptr, config)
        : render_gpu(scene_ptr, config);

    printf("Done rendering\n");

    destroy_Scene(scene_ptr);
    save_to_file("outputs/out.png", render_result);
    destroy_Texture(render_result);
    return 0;
}