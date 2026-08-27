# nbody: CUDA vs HIP

Two standalone builds of the classic all-pairs gravitational n-body
simulation, side by side:

- **[`nbody_cuda/`](nbody_cuda/)** — the original NVIDIA `cuda-samples`
  `nbody` sample, unmodified algorithm, made to build on its own (no longer
  depends on a full `cuda-samples` checkout). Targets NVIDIA GPUs.
- **[`nbody_hip/`](nbody_hip/)** — a HIP port of the same sample, for AMD
  GPUs. Headless (no OpenGL/GLUT) — see
  [`nbody_hip/CUDA_TO_HIP_PORTING_NOTES.md`](nbody_hip/CUDA_TO_HIP_PORTING_NOTES.md)
  for exactly what changed and why.

Both were benchmarked against each other: 1–5× AMD Instinct MI210 vs.
1–8× NVIDIA A100-SXM4-80GB, same problem size, same algorithm. See
[Benchmark results](#benchmark-results) below.

## Prerequisites

| | `nbody_cuda` | `nbody_hip` |
|---|---|---|
| Toolchain | CUDA Toolkit (`nvcc`) | ROCm (`hipcc`) |
| GPU | NVIDIA (tested: A100, SM 8.0) | AMD (tested: MI210, gfx90a) |
| Display / GL | Needed for the interactive GUI; `-benchmark`/`-compare`/`-cpu` modes work headless too | Not needed — this build is headless-only by design |

## Build & run — CUDA

```sh
cd nbody_cuda
cmake -B build
cmake --build build -j
./build/nbody --help
```

```
> Command line options
    -fullscreen       (run n-body simulation in fullscreen mode)
    -fp64             (use double precision floating point values for simulation)
    -hostmem          (stores simulation data in host memory)
    -benchmark        (run benchmark to measure performance)
    -numbodies=<N>    (number of bodies (>= 1) to run in simulation)
    -device=<d>       (where d=0,1,2.... for the CUDA device to use)
    -numdevices=<i>   (where i=(number of CUDA devices > 0) to use for simulation)
    -compare          (compares simulation results running once on the default GPU and once on the CPU)
    -cpu              (run n-body simulation on the CPU)
    -tipsy=<file.bin> (load a tipsy model file for simulation)
```

Headless example (no display needed):

```sh
./build/nbody -benchmark -numdevices=1 -numbodies=2097152 -fp64
```

## Build & run — HIP

```sh
cd nbody_hip
cmake -B build -DCMAKE_CXX_COMPILER=hipcc -DGPU_TARGETS=gfx90a   # set GPU_TARGETS for your GPU
cmake --build build -j
./build/nbody_hip --help
```

`GPU_TARGETS` defaults to `gfx90a` (MI210) — override for other AMD GPUs,
e.g. `-DGPU_TARGETS=gfx942` for MI300.

`nbody_hip`'s CLI deliberately matches `nbody`'s flag-for-flag (same table
as above), since there's no GUI in this build to fall back to, running with
neither `-benchmark` nor `-compare` benchmarks by default. Full differences
are in `nbody_hip/README.md`.

```sh
./build/nbody_hip -benchmark -numdevices=1 -numbodies=2097152 -fp64
```

Sanity-check any build against the CPU reference implementation before
trusting its numbers:

```sh
./build/nbody[_hip] -numbodies=65536 -compare
```

## Multi-GPU benchmark scripts

Both `nbody_cuda/bench/` and `nbody_hip/bench/` ship the same set of
scripts, one per GPU count, each writing its own timestamped log to
`bench/logs/`:

```sh
cd nbody_cuda/bench   # or nbody_hip/bench
./bench_1gpu.sh       # -numdevices=1
./bench_2gpu.sh       # -numdevices=2
./bench_4gpu.sh       # -numdevices=4
./bench_5gpu.sh       # -numdevices=5
./bench_8gpu.sh       # -numdevices=8 ( only for A100, we don't have 8 MI210 gpu cards )
./run_all.sh          # runs the full sweep back-to-back, then prints a scaling table
./summarize.sh         # re-prints that table from the latest logs, any time
```

All default to `-numbodies=2097152 -fp64`; override with env vars, e.g.
`NUMBODIES=1048576 ./bench_4gpu.sh`, or pass extra flags through, e.g.
`./bench_1gpu.sh -i=30`. `NBODY_BIN` overrides the binary path if you built
somewhere other than `../build/nbody[_hip]`.

## Benchmark results

fp64, N = 2,097,152 bodies, 10 iterations, direct O(N²) kernel (no
Tensor/Matrix cores on either side):

| GPUs | MI210 total | MI210 per-card | MI210 efficiency | A100 total | A100 per-card | A100 efficiency |
|---|---|---|---|---|---|---|
| 1 | 9,496 GFLOP/s | 9,496 | 100.0% (baseline) | 7,638 GFLOP/s | 7,638 | 100.0% (baseline) |
| 2 | 17,729 GFLOP/s | 8,865 | 93.3% | 15,081 GFLOP/s | 7,540 | 98.7% |
| 4 | 35,153 GFLOP/s | 8,788 | 92.5% | 30,028 GFLOP/s | 7,507 | 98.3% |
| 5 | 43,212 GFLOP/s | 8,642 | 91.0% | 35,652 GFLOP/s | 7,130 | 93.3% |
| 8 | *(only 5 MI210s in that node)* | | | 56,743 GFLOP/s | 7,093 | 92.9% |

Both scale well from a genuine single-GPU baseline. The MI210 delivers
higher raw per-card throughput on this unoptimized kernel (~8.6–9.5 vs.
~7.1–7.6 TFLOP/s), while the A100 cluster holds tighter multi-GPU
efficiency (NVLink/NVSwitch vs. plain PCIe P2P here). Neither number should
be read as each vendor's ceiling — this kernel wasn't retuned for either
architecture.

## Repo layout notes

- Each of `nbody_cuda/` and `nbody_hip/` is fully standalone: their
  `Common/` vendors the handful of `cuda-samples` helper headers each needs
  (`helper_cuda.h`, `helper_gl.h`, etc.), so neither depends on a full
  `cuda-samples` checkout existing anywhere else.
- `nbody_cuda/README.md` is the original NVIDIA sample description.
  `nbody_hip/README.md` and `nbody_hip/CUDA_TO_HIP_PORTING_NOTES.md`
  document the port itself — read the porting notes if you're bringing this
  same sample to another AMD/ROCm target, they list every place the
  automated `hipify-perl` conversion needed a manual fix.
- `*.prehip` files in `nbody_hip/` are `hipify-perl`'s own automatic
  pre-conversion backups — harmless, safe to ignore or delete.
- Neither `build/` directory is checked in (see `.gitignore`) — CMake build
  directories embed absolute paths and are not portable between clones or
  machines; always `cmake -B build` fresh rather than copying one over.
