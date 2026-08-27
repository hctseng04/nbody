cmake_minimum_required(VERSION 3.20)

# Headless HIP port of the CUDA nbody sample. Configure with hipcc as the
# C++ compiler, e.g.:
#   cmake -B build -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_HIP_ARCHITECTURES=gfx90a
#   cmake --build build
#
# CMAKE_HIP_ARCHITECTURES defaults to gfx90a (MI210); override for other GPUs.

project(nbody_hip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED GPU_TARGETS)
    set(GPU_TARGETS "gfx90a" CACHE STRING "AMD GPU target architecture(s)")
endif()

# This build intentionally excludes OpenGL/GLUT (see README.md): it covers
# the compute (BodySystemHIP / BodySystemCPU) path only, not the interactive
# visualization from the original CUDA sample.
add_executable(nbody_hip
    nbody_hip_bench.cpp
    bodysystemhip.cpp
)

target_compile_definitions(nbody_hip PRIVATE NBODY_NO_GL)

target_include_directories(nbody_hip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Common
)

target_compile_options(nbody_hip PRIVATE --offload-arch=${GPU_TARGETS})
target_link_options(nbody_hip PRIVATE --offload-arch=${GPU_TARGETS})
