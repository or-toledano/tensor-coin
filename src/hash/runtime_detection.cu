// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
// based on:
// stackoverflow.com/questions/12828468/detecting-nvidia-gpus-without-cuda

#include <iostream>
#include <cuda.h>

#ifdef WINDOWS
#include <Windows.h>
#else

#include <dlfcn.h>

#endif

void *loadCudaLibrary() {
#ifdef WINDOWS
    return LoadLibraryA("nvcuda.dll");
#else
    return dlopen("libcuda.so", RTLD_NOW);
#endif
}

void (*getProcAddress(void *lib, const char *name))(void) {
#ifdef WINDOWS
    return (void (*)(void))GetProcAddress(lib, name);
#else
    return (void (*)(void)) dlsym(lib, (const char *) name);
#endif
}

int freeLibrary(void *lib) {
#ifdef WINDOWS
    return FreeLibrary(lib);
#else
    return dlclose(lib);
#endif
}

typedef CUresult CUDAAPI (*cuInit_pt)(unsigned int Flags);

typedef CUresult CUDAAPI (*cuDeviceGetCount_pt)(int *count);

typedef CUresult CUDAAPI (*cuDeviceComputeCapability_pt)(int *major, int *minor,
                                                         CUdevice dev);

int detect_cuda() {
    std::clog << "Searching for a CUDA device ...\n";
    void *cuLib;
    cuInit_pt my_cuInit = NULL;
    cuDeviceGetCount_pt my_cuDeviceGetCount = NULL;
    cuDeviceComputeCapability_pt my_cuDeviceComputeCapability = NULL;

    if ((cuLib = loadCudaLibrary()) == NULL) {
        std::clog << "CUDA library can't be loaded";
        return 1;
    }

    if ((my_cuInit = (cuInit_pt) getProcAddress(cuLib, "cuInit"))
        == NULL) {
        std::clog << "cuInit not found";
        return 1;
    }
    if ((my_cuDeviceGetCount = (cuDeviceGetCount_pt) getProcAddress(
            cuLib, "cuDeviceGetCount")) == NULL) {
        std::clog << "cuDeviceGetCount not found";
        return 1;
    }
    if ((my_cuDeviceComputeCapability = (cuDeviceComputeCapability_pt)
            getProcAddress(cuLib, "cuDeviceComputeCapability")) == NULL) {
        std::clog << "cuDeviceComputeCapability not found";
        return 1;
    }

    {
        int count, i;
        if (CUDA_SUCCESS != my_cuInit(0)) {
            std::clog << "cuInit failed";
            return 1;
        }
        if (CUDA_SUCCESS != my_cuDeviceGetCount(&count)) {
            std::clog << "cuInit failed";
            return 1;
        }

        for (i = 0; i < count; i++) {
            int major, minor;
            if (CUDA_SUCCESS !=
                my_cuDeviceComputeCapability(&major, &minor, i)) {
                std::clog << "cuDeviceComputeCapability failed";
                return 1;
            }
            std::clog << "dev " << i << " CUDA compute capability major "
                      << major << " minor " << minor << "\n";
        }
    }
    freeLibrary(cuLib);
    return 0;
}

