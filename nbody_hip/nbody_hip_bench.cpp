/*
 * Headless (no OpenGL/GLUT) HIP n-body benchmark.
 *
 * This is a compact, display-free driver for BodySystemHIP, ported from the
 * -benchmark / -compare paths of the original CUDA nbody.cpp sample -- see
 * CUDA_TO_HIP_PORTING_NOTES.md, section 8, for why the interactive
 * OpenGL/GLUT visualization itself isn't built here. Command-line handling
 * intentionally mirrors the original sample's `-flag`/`-flag=value` syntax
 * (via the same Common/helper_string.h parsing helpers the CUDA sample
 * uses) and reproduces its device-selection / multi-GPU / default-numbodies
 * logic, so this tool's `--help` matches the CUDA tool's flag-for-flag,
 * modulo the GUI-only options noted below.
 *
 * Differences from the CUDA sample's CLI, all because this build has no
 * GL/GLUT and therefore no windowed mode at all:
 *   -fullscreen   accepted but ignored (nothing to make fullscreen)
 *   -benchmark    accepted but not required -- with neither -benchmark nor
 *                 -compare given, this tool runs the benchmark anyway
 *                 (the CUDA sample would instead open its GUI)
 */

#include <helper_cuda.h>  // checkCudaErrors(), getLastCudaError(), and (via
                          // helper_string.h) checkCmdLineFlag() / getCmdLineArgumentInt() /
                          // getCmdLineArgumentString()
#include <helper_timer.h> // StopWatchInterface, sdkCreateTimer() et al. -- used for the
                          // -cpu timing path, exactly as nbody.cpp's _runBenchmark() does

#include <cstdio>
#include <cstdlib>
#include <string>

#include "bodysystemcpu.h"
#include "bodysystemhip.h"

namespace {

// Same default single-demo parameters as demoParams[0] in nbody.cpp.
constexpr float kClusterScale  = 1.54f;
constexpr float kVelocityScale = 8.0f;
constexpr float kSoftening     = 0.1f;
constexpr float kDamping       = 1.0f;
constexpr float kTimestep      = 0.016f;
constexpr float kTolerance     = 0.0005f; // matches nbody.cpp's _compareResults()

void printHelp(const char *argv0)
{
    printf("> Command line options\n");
    printf("    -fullscreen       (ignored: this is a headless, GL-free build)\n");
    printf("    -fp64             (use double precision floating point values for simulation)\n");
    printf("    -hostmem          (stores simulation data in host memory)\n");
    printf("    -benchmark        (run benchmark to measure performance)\n");
    printf("    -numbodies=<N>    (number of bodies (>= 1) to run in simulation)\n");
    printf("    -device=<d>       (where d=0,1,2.... for the HIP device to use)\n");
    printf("    -numdevices=<i>   (where i=(number of HIP devices > 0) to use for simulation)\n");
    printf("    -compare          (compares simulation results running once on the default GPU and once on the CPU)\n");
    printf("    -cpu              (run n-body simulation on the CPU)\n");
    printf("    -tipsy=<file.bin> (load a tipsy model file for simulation)\n");
    printf("    -i=<N>            (number of benchmark iterations; default 10)\n");
    printf("    -blockSize=<N>    (HIP block size; default 256)\n");
    printf("Usage: %s [options above]\n", argv0);
}

int flopsPerInteraction = 20;

void computePerfStats(int numBodies, double &interactionsPerSecond, double &gflops, float milliseconds, int iterations)
{
    interactionsPerSecond = (double)numBodies * (double)numBodies;
    interactionsPerSecond *= 1e-9 * iterations * 1000 / milliseconds;
    gflops = interactionsPerSecond * (double)flopsPerInteraction;
}

// One-shot GPU-vs-CPU cross check, ported from NBodyDemo<T>::_compareResults()
// in nbody.cpp. `nbody` has already had one update() applied by the caller;
// hPos/hVel are the *pre-update* initial conditions.
template <typename T>
bool compareResults(BodySystem<T> *nbody, const T *hPos, const T *hVel, int numBodies)
{
    BodySystemCPU<T> nbodyCpu(numBodies);
    nbodyCpu.setArray(BODYSYSTEM_POSITION, hPos);
    nbodyCpu.setArray(BODYSYSTEM_VELOCITY, hVel);
    nbodyCpu.setSoftening(kSoftening);
    nbodyCpu.setDamping(kDamping);
    nbodyCpu.update(kTimestep);

    T   *gpuPos = nbody->getArray(BODYSYSTEM_POSITION);
    T   *cpuPos = nbodyCpu.getArray(BODYSYSTEM_POSITION);
    bool passed = true;

    for (int i = 0; i < numBodies * 4; i++) {
        if (fabs((double)(cpuPos[i] - gpuPos[i])) > kTolerance) {
            passed = false;
            printf("Error: (host)%f != (device)%f\n", (double)cpuPos[i], (double)gpuPos[i]);
        }
    }

    printf(passed ? "  OK\n" : "  FAILED\n");
    return passed;
}

// Timed iteration loop + perf report, ported from NBodyDemo<T>::_runBenchmark().
// `system` doubles as the un-typed handle used for hipEvent-based timing (GPU
// path) vs. StopWatchInterface (CPU path).
template <typename T> void runBenchmarkLoop(BodySystem<T> *nbody, bool useCpu, int numBodies, int numIterations)
{
    // once without timing to prime the device / caches
    nbody->update(kTimestep);

    float               milliseconds = 0;
    hipEvent_t          startEvent = nullptr, stopEvent = nullptr;
    StopWatchInterface *timer      = nullptr;

    if (useCpu) {
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);
    }
    else {
        checkCudaErrors(hipEventCreate(&startEvent));
        checkCudaErrors(hipEventCreate(&stopEvent));
        checkCudaErrors(hipEventRecord(startEvent, 0));
    }

    for (int i = 0; i < numIterations; ++i) {
        nbody->update(kTimestep);
    }

    if (useCpu) {
        sdkStopTimer(&timer);
        milliseconds = sdkGetTimerValue(&timer);
        sdkDeleteTimer(&timer);
    }
    else {
        checkCudaErrors(hipEventRecord(stopEvent, 0));
        checkCudaErrors(hipEventSynchronize(stopEvent));
        checkCudaErrors(hipEventElapsedTime(&milliseconds, startEvent, stopEvent));
        checkCudaErrors(hipEventDestroy(startEvent));
        checkCudaErrors(hipEventDestroy(stopEvent));
    }

    double interactionsPerSecond = 0, gflops = 0;
    computePerfStats(numBodies, interactionsPerSecond, gflops, milliseconds, numIterations);

    printf("%d bodies, total time for %d iterations: %.3f ms\n", numBodies, numIterations, milliseconds);
    printf("= %.3f billion interactions per second\n", interactionsPerSecond);
    printf("= %.3f %s-precision GFLOP/s at %d flops per interaction\n",
           gflops,
           (sizeof(T) > 4) ? "double" : "single",
           flopsPerInteraction);
}

struct Options
{
    bool        fullscreen         = false;
    bool        fp64               = false;
    bool        useHostMem         = false;
    bool        benchmark          = false;
    bool        compareToCPU       = false;
    bool        useCpu             = false;
    int         numBodies          = 0;
    bool        numBodiesExplicit  = false;
    int         deviceId           = 0;
    bool        deviceExplicit     = false;
    int         numDevsRequested   = 1;
    bool        numDevsExplicit    = false;
    int         numIterations      = 0;
    int         blockSize          = 0;
    std::string tipsyFile;
};

// Runs the whole flow for one floating-point type T -- device/CPU pick,
// body-count defaulting, initial conditions, then -compare and/or
// -benchmark (or, lacking a GUI to fall back to, -benchmark by default).
// Returns the process exit code.
template <typename T> int run(const Options &opt)
{
    flopsPerInteraction = (sizeof(T) > 4) ? 30 : 20;

    int         numBodies = opt.numBodies;
    int         devID     = opt.deviceId;
    bool        useP2P    = true;
    bool        useHostMem = opt.useHostMem || opt.useCpu; // matches nbody.cpp: useCpu implies useHostMem
    hipDeviceProp_t props{};

    if (!opt.useCpu) {
        checkCudaErrors(hipSetDevice(devID));
        checkCudaErrors(hipGetDeviceProperties(&props, devID));
        printf("> HIP device [%d]: %s (%s)\n", devID, props.name, props.gcnArchName);

        if (opt.numDevsRequested > 1 && !useHostMem) {
            // Mirrors nbody.cpp: fall back to host memory if any requested
            // GPU beyond #0 can't peer-access GPU #0.
            bool allSupportP2P = true;
            for (int i = 1; i < opt.numDevsRequested; ++i) {
                int canAccessPeer = 0;
                checkCudaErrors(hipDeviceCanAccessPeer(&canAccessPeer, i, 0));
                if (!canAccessPeer) allSupportP2P = false;
            }
            if (!allSupportP2P) {
                useHostMem = true;
                useP2P     = false;
                printf("> Not all requested devices support P2P -- falling back to host memory\n");
            }
        }

        if (!opt.numBodiesExplicit) {
            if (opt.numDevsRequested == 1) {
                numBodies = opt.compareToCPU ? 4096 : opt.blockSize * 4 * props.multiProcessorCount;
            }
            else {
                numBodies = 0;
                for (int i = 0; i < opt.numDevsRequested; i++) {
                    hipDeviceProp_t p;
                    checkCudaErrors(hipGetDeviceProperties(&p, i));
                    numBodies += opt.blockSize * 4 * p.multiProcessorCount;
                }
            }
        }
    }

    printf("> %s mode\n", opt.fullscreen ? "Fullscreen (ignored)" : "Headless");
    printf("> Simulation data stored in %s memory\n", useHostMem ? "system" : "video");
    printf("> %s precision floating point simulation\n", opt.fp64 ? "Double" : "Single");
    printf("> %d device(s) used for simulation\n", opt.useCpu ? 0 : opt.numDevsRequested);
    printf("> %d bodies\n", numBodies);

    BodySystemCPU<T> *cpuSystem = nullptr;
    BodySystemHIP<T> *gpuSystem = nullptr;
    BodySystem<T>    *nbody     = nullptr;

    if (opt.useCpu) {
        cpuSystem = new BodySystemCPU<T>(numBodies);
        nbody     = cpuSystem;
    }
    else {
        gpuSystem = new BodySystemHIP<T>(
            numBodies, opt.numDevsRequested, opt.blockSize, /*usePBO=*/false, useHostMem, useP2P, devID);
        nbody = gpuSystem;
    }

    nbody->setSoftening(kSoftening);
    nbody->setDamping(kDamping);

    T    *hPos   = nullptr;
    T    *hVel   = nullptr;
    float *hColor = nullptr;

    if (!opt.tipsyFile.empty()) {
        nbody->loadTipsyFile(opt.tipsyFile);
        numBodies = nbody->getNumBodies();
        printf("> Loaded %d bodies from tipsy file \"%s\"\n", numBodies, opt.tipsyFile.c_str());
    }
    else {
        hPos   = new T[numBodies * 4];
        hVel   = new T[numBodies * 4];
        hColor = new float[numBodies * 4];
        randomizeBodies(NBODY_CONFIG_SHELL, hPos, hVel, hColor, kClusterScale, kVelocityScale, numBodies, true);
        nbody->setArray(BODYSYSTEM_POSITION, hPos);
        nbody->setArray(BODYSYSTEM_VELOCITY, hVel);
    }

    int  exitCode  = EXIT_SUCCESS;
    bool ranCompare = false;

    // Same precedence as nbody.cpp's main(): -benchmark wins over -compare.
    // (-cpu forces -compare off already, in main().) With neither -benchmark
    // nor -compare given, this headless build has no GUI to fall back to, so
    // it benchmarks anyway.
    if (opt.compareToCPU && !opt.benchmark) {
        if (!hPos) {
            printf("-compare is not supported together with -tipsy (no host-side reference data kept); "
                   "running the benchmark instead.\n");
        }
        else {
            nbody->update(kTimestep);
            bool passed = compareResults(nbody, hPos, hVel, numBodies);
            exitCode    = passed ? EXIT_SUCCESS : EXIT_FAILURE;
            ranCompare  = true;
        }
    }

    if (!ranCompare) {
        int iterations = opt.numIterations > 0 ? opt.numIterations : 10;
        runBenchmarkLoop(nbody, opt.useCpu, numBodies, iterations);
    }

    delete cpuSystem;
    delete gpuSystem;
    delete[] hPos;
    delete[] hVel;
    delete[] hColor;

    return exitCode;
}

} // namespace

int main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "h")) {
        printHelp(argv[0]);
        return EXIT_SUCCESS;
    }

    Options opt;
    opt.fullscreen   = checkCmdLineFlag(argc, (const char **)argv, "fullscreen");
    opt.fp64         = checkCmdLineFlag(argc, (const char **)argv, "fp64");
    opt.useHostMem   = checkCmdLineFlag(argc, (const char **)argv, "hostmem");
    opt.benchmark    = checkCmdLineFlag(argc, (const char **)argv, "benchmark");
    opt.compareToCPU = checkCmdLineFlag(argc, (const char **)argv, "compare");
    opt.useCpu       = checkCmdLineFlag(argc, (const char **)argv, "cpu");

    opt.deviceExplicit = checkCmdLineFlag(argc, (const char **)argv, "device");
    if (opt.deviceExplicit) opt.deviceId = getCmdLineArgumentInt(argc, (const char **)argv, "device");

    opt.numDevsExplicit = checkCmdLineFlag(argc, (const char **)argv, "numdevices");
    if (opt.numDevsExplicit) opt.numDevsRequested = getCmdLineArgumentInt(argc, (const char **)argv, "numdevices");

    if (opt.numDevsRequested > 1 && opt.deviceExplicit) {
        fprintf(stderr, "You can't use -numdevices and -device at the same time.\n");
        return EXIT_FAILURE;
    }

    if (!opt.useCpu) {
        int numDevsAvailable = 0;
        hipGetDeviceCount(&numDevsAvailable);
        if (numDevsAvailable < opt.numDevsRequested) {
            fprintf(stderr, "Error: only %d HIP device(s) available, %d requested.\n", numDevsAvailable, opt.numDevsRequested);
            return EXIT_FAILURE;
        }
    }

    if (opt.useCpu && opt.compareToCPU) {
        // matches nbody.cpp: useCpu forces compareToCPU off (nothing to compare against)
        printf("-compare has no effect together with -cpu; ignoring, running the benchmark instead.\n");
        opt.compareToCPU = false;
    }

    opt.blockSize = checkCmdLineFlag(argc, (const char **)argv, "blockSize")
                        ? getCmdLineArgumentInt(argc, (const char **)argv, "blockSize")
                        : 256;
    opt.numIterations = checkCmdLineFlag(argc, (const char **)argv, "i")
                             ? getCmdLineArgumentInt(argc, (const char **)argv, "i")
                             : 0;

    opt.numBodiesExplicit = checkCmdLineFlag(argc, (const char **)argv, "numbodies");
    if (opt.numBodiesExplicit) {
        opt.numBodies = getCmdLineArgumentInt(argc, (const char **)argv, "numbodies");
        if (opt.numBodies < 1) {
            fprintf(stderr, "Error: \"number of bodies\" specified %d is invalid. Value should be >= 1\n", opt.numBodies);
            return EXIT_FAILURE;
        }
        if (opt.numBodies % opt.blockSize) {
            int rounded = ((opt.numBodies / opt.blockSize) + 1) * opt.blockSize;
            printf("Warning: \"number of bodies\" specified %d is not a multiple of %d.\n", opt.numBodies, opt.blockSize);
            printf("Rounding up to the nearest multiple: %d.\n", rounded);
            opt.numBodies = rounded;
        }
    }
    else if (opt.useCpu) {
        opt.numBodies = 4096;
    }
    // else: defaulted later in run<T>(), once the device's SM count is known.

    char *tipsyArg = nullptr;
    if (getCmdLineArgumentString(argc, (const char **)argv, "tipsy", &tipsyArg) && tipsyArg) {
        opt.tipsyFile = tipsyArg;
    }

    if (opt.fp64) {
        return run<double>(opt);
    }
    return run<float>(opt);
}
