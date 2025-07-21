cmake_minimum_required(VERSION 3.31)
project(GravitationalRayMarching CUDA C CXX)

set(CMAKE_CUDA_STANDARD 20)
add_library(GravitationalRayMarching STATIC)
target_include_directories(GravitationalRayMarching PUBLIC lib/include)
target_include_directories(GravitationalRayMarching PUBLIC lib/CImg)
file(GLOB LIB_SOURCES CONFIGURE_DEPENDS
        "lib/src/*.cu"
        "lib/src/api/*.cu"
)
target_sources(GravitationalRayMarching PRIVATE ${LIB_SOURCES})
target_compile_options(GravitationalRayMarching PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -Wno-deprecated-gpu-targets
        >)

add_executable(GravitationalRayMarchingMain main.cu)
target_include_directories(GravitationalRayMarchingMain PUBLIC lib/include)
target_link_libraries(GravitationalRayMarchingMain PUBLIC GravitationalRayMarching)

add_executable(GravitationalRayMarchingMainC main.c)
target_include_directories(GravitationalRayMarchingMainC PUBLIC lib/include/api)
target_link_libraries(GravitationalRayMarchingMainC PUBLIC GravitationalRayMarching)

# move assets dir to out dir
add_custom_command(TARGET GravitationalRayMarchingMain POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets/ $<TARGET_FILE_DIR:GravitationalRayMarchingMain>/assets)

# move assets dir to out dir
add_custom_command(TARGET GravitationalRayMarchingMainC POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets/ $<TARGET_FILE_DIR:GravitationalRayMarchingMainC>/assets)

set_target_properties(GravitationalRayMarching PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CUDA)

set_target_properties(GravitationalRayMarchingMain PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CUDA)

add_custom_target( clear_rendered_output_frames
        COMMAND rm -fr outputs
        WORKING_DIRECTORY $<TARGET_FILE_DIR:GravitationalRayMarchingMain>
        COMMENT "clear previously rendered frames"
)

add_custom_target( render_frames_gpu
        COMMAND GravitationalRayMarchingMain gpu 128 64 true
        WORKING_DIRECTORY $<TARGET_FILE_DIR:GravitationalRayMarchingMain>
        COMMENT "render frames on gpu and place in outputs dir"
)

add_custom_target( merge_frames
        COMMAND ffmpeg -framerate 10 -stream_loop 4 -i "outputs/out_%03d.png" -c:v libx264 -r 20 -y outputs/out.mp4
        WORKING_DIRECTORY $<TARGET_FILE_DIR:GravitationalRayMarchingMain>
        COMMENT "combine output frames into an mp4"
)

add_dependencies( render_frames_gpu clear_rendered_output_frames )
add_dependencies( merge_frames render_frames_gpu )