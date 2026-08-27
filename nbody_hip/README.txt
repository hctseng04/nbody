cmake -B build -DCMAKE_CXX_COMPILER=hipcc -DGPU_TARGETS=gfx90a

cmake --build build -j
