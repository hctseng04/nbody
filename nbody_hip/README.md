# nbody — HIP port (AMD MI210)

This is a HIP port of `cpp/5_Domain_Specific/nbody`, built and verified against
two AMD Instinct MI210 GPUs (gfx90a) on this machine with ROCm 7.14
(`hipcc`, `hipify-perl`).

## What was ported

* `bodysystemcuda.cu` → `bodysystemhip.cpp` (device kernels + host launch code)
* `bodysystemcuda.h` / `bodysystemcuda_impl.h` → `bodysystemhip.h` /
  `bodysystemhip_impl.h`, class `BodySystemCUDA` → `BodySystemHIP`
* `Common/helper_cuda.h` and friends copied locally into `Common/` and
  hipified (kept local, rather than editing the shared `../../../Common`,
  so the original CUDA samples tree is untouched)
* All `cuda*` API calls, types, and error-check macros converted to their
  `hip*` equivalents (via `hipify-perl -inplace`, plus the manual fixes
  below that hipify-perl on this ROCm version did not catch)

## New file: `nbody_hip_bench.cpp` (headless benchmark)

The original `nbody.cpp` interleaves OpenGL/GLUT calls throughout `main()`,
`display()`, `initGL()`, the mouse/keyboard callbacks, etc. — even its
`-benchmark` mode still compiles that GL/GLUT code, it just skips *calling*
it at runtime. Porting that visualization path needs:

1. `freeglut-devel` / `mesa-libGL-devel` headers (only the **runtime** libs
   are installed on this node — `freeglut-3.2.1` and `libGLEW-2.2.0` — the
   `-devel` packages with headers are not, and installing them needs root),
   and
2. a real GLX/EGL display (this node has no `DISPLAY` and no Xvfb), and
3. HIP's OpenGL-interop API is a subset of CUDA's —
   `cudaGraphicsResourceSetMapFlags()` (used to hint read-only/write-discard
   access per frame) has **no HIP equivalent at all**; it had to be dropped
   (see comments in `bodysystemhip.cpp` / `bodysystemhip_impl.h`).

None of that is fixable from inside this repo/session, so this port ships a
new, compact, GL-free driver, `nbody_hip_bench.cpp`, that exercises the same
`BodySystemHIP` / `BodySystemCPU` classes the real app uses, reusing the
`-benchmark` math (`computePerfStats`) and CPU-vs-GPU check
(`_compareResults`) from `nbody.cpp`. It's the practical target for a
headless GPU compute node anyway. `render_particles.{h,cpp}` and `nbody.cpp`
are carried over unmodified for reference / a future GUI build on a machine
that has the GL dev headers and a display — see "Rebuilding the interactive
GUI" below.

## Manual fixes beyond `hipify-perl`

`hipify-perl -inplace *.cpp *.h Common/*.h` handled the bulk of the
`cuda*` → `hip*` renaming, but left a few things that needed hand fixing:

1. **`#include <helper_cuda.h>` was rewritten to
   `#include <hip/hip_runtime_api.h>`.** hipify-perl treats `helper_cuda.h`
   as a known NVIDIA-samples header and always redirects it — even though
   here it's *our own* file (copied into `Common/`, and hipified in place)
   that defines `checkCudaErrors()`/`getLastCudaError()`. Reverted the 5
   affected `#include` lines back to `<helper_cuda.h>`.
2. **`checkCudaErrors()` / `getLastCudaError()` / `_cudaGetErrorEnum()`
   silently compiled away to nothing.** They're guarded by
   `#ifdef __DRIVER_TYPES_H__` / `#ifdef __CUDA_RUNTIME_H__` in
   `Common/helper_cuda.h` — CUDA-header include guards that HIP headers
   don't define. Widened those 3 guards to
   `#if defined(...) || defined(__HIPCC__)`.
3. **`cudaGraphicsResourceSetMapFlags()` and
   `cudaGraphicsMapFlags{ReadOnly,WriteDiscard,None}`** were not in
   hipify-perl's translation table on this ROCm version (unlike
   `cudaGraphicsRegisterBuffer`/`Map`/`UnmapResources`, which were). As noted
   above, HIP has no equivalent call at all; the per-map flag hint calls
   were removed (registration already passes
   `hipGraphicsRegisterFlagsNone`), documented in code.
4. `cuda_gl_interop.h` → `hip/hip_gl_interop.h` (present in this ROCm
   install; not in hipify-perl's table either).
5. `tipsy.h` used `cout`/`cerr`/`endl` under `using namespace std;` without
   `#include <iostream>` — a latent bug in the original sample that
   surfaces once you don't happen to pull in `<iostream>` transitively.
   Added the missing include.
6. `#include <GL/freeglut.h>` in `bodysystemhip.cpp` was vestigial (no
   GLUT symbol is actually used in that file) — gated it, and the
   PBO/GL-interop code blocks in `bodysystemhip_impl.h`, behind a new
   `NBODY_NO_GL` macro so the compute path builds without GLUT headers at
   all. `usePBO=true` still requires a GL build (unchanged behavior when
   `NBODY_NO_GL` isn't defined).

## Build

```sh
cd cpp/5_Domain_Specific/nbody_hip
cmake -B build -DCMAKE_CXX_COMPILER=hipcc -DGPU_TARGETS=gfx90a
cmake --build build -j
```

(or directly: `hipcc -std=c++17 -DNBODY_NO_GL --offload-arch=gfx90a -I. -ICommon -O3 nbody_hip_bench.cpp bodysystemhip.cpp -o nbody_hip_bench`)

## Run

```sh
./build/nbody_hip_bench -numbodies=65536 -i=30 -compare
```

`nbody_hip_bench --help` prints the flag reference. Its CLI is a
deliberate flag-for-flag match of the CUDA sample's own `--help`
(`-fullscreen -fp64 -hostmem -benchmark -numbodies=<N> -device=<d>
-numdevices=<i> -compare -cpu -tipsy=<file.bin>`, plus the CUDA sample's
undocumented `-i=<N>`/`-blockSize=<N>`), including its device-selection,
multi-GPU P2P-fallback, and default-`numbodies` logic — see
`CUDA_TO_HIP_PORTING_NOTES.md` §12 for the one-by-one mapping and the
handful of differences forced by having no GUI at all (`-fullscreen` is a
no-op; `-benchmark` is implied when neither it nor `-compare` is given,
since there's no windowed mode to fall back to otherwise).

Measured on this node (MI210, gfx90a), N=196608, 30 iterations:

| precision | GFLOP/s  |
|-----------|----------|
| fp32      | ~8,575   |
| fp64      | ~9,158   |

## Rebuilding the interactive GUI (not done here)

To get the original windowed/animated `nbody` back, on a machine with a
display attached to the MI210:

1. Install GL/GLUT dev headers: `freeglut-devel`, `mesa-libGL-devel`,
   `glew-devel` (or equivalent for your distro).
2. Build `nbody.cpp` + `render_particles.cpp` + `bodysystemhip.cpp`
   *without* `-DNBODY_NO_GL`.
3. Fix up the same `cuda_gl_interop.h`/`GL/freeglut.h` and
   `checkCudaErrors` include issues described above in `nbody.cpp` and
   `render_particles.cpp` (they weren't touched here since they're
   unreachable without GL headers to begin with).
4. Note the `cudaGraphicsResourceSetMapFlags` gap: the PBO ping-pong will
   still work correctly (registration flag is set once at
   `hipGraphicsGLRegisterBuffer` time), just without CUDA's per-frame
   read-only/write-discard access hint.
