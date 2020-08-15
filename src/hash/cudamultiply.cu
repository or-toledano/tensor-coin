// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#include "cudamultiply.cuh"
#include "uhash.hpp"
#include <cublas_v2.h>
#include <iostream>
#include <string>


using std::cerr;

cublasHandle_t handle;
float *devPtrA, *devPtrB, *devPtrRes;

void myCudaFree() {
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrRes);
    cublasDestroy(handle);
}


void myCudaCondDtor(cublasStatus_t stat, const char *msg) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << msg << " failed\n";
        myCudaFree();
        exit(EXIT_FAILURE);
    }
}

// This is so sad. Can we please get integer MMU to CUDA ?? (cublasSgemmEx
// does support INT8 for A/B, but not for C!)
// TODO: change algo so it will reinterpret char as float to avoid conversion ??
void uchar_to_float(const unsigned char *src, float *dst) {
    for (int i = 0; i < MAT_SIZE; ++i)
        dst[i] = static_cast<float>(src[i]);
}

// Multiply the ind0 matrix with the ind1 matrix, add to the indRes matrix
// assume all indices are different and matrices are initialized
void
cudaMultiplyAdd(const unsigned char *cubeSrc, unsigned char *cubeDst, int
ind0, int ind1, int indRes) {
    // Error codes for alloc on gpu
    cudaError_t cudaStat0 = cudaMalloc((float **) &devPtrA,
                                       MAT_SIZE * sizeof(float));
    cudaError_t cudaStat1 = cudaMalloc((float **) &devPtrB,
                                       MAT_SIZE * sizeof(float));
    cudaError_t cudaStat2 = cudaMalloc((float **) &devPtrRes,
                                       MAT_SIZE * sizeof(float));
    if (cudaStat0 != cudaSuccess || cudaStat1 != cudaSuccess ||
        cudaStat2 != cudaSuccess) {
        cerr << "device memory allocation" << " failed\n";
        myCudaFree();
        exit(EXIT_FAILURE);
    }
    cublasStatus_t stat = cublasCreate(&handle);
    // Error codes for CUBLAS initialization
    myCudaCondDtor(stat, "CuBLAS initialization");



    // Download from cpu to gpu

    float tmp[MAT_SIZE];
    uchar_to_float(cubeSrc + MAT_SIZE * ind0, tmp);
    stat = cublasSetMatrix(DIM, DIM, sizeof(float), tmp, DIM, devPtrA, DIM);
    // Error codes for cpu to gpu bus
    myCudaCondDtor(stat, "devPtrA download");

    uchar_to_float(cubeSrc + MAT_SIZE * ind1, tmp);
    stat = cublasSetMatrix(DIM, DIM, sizeof(float), tmp, DIM, devPtrB, DIM);
    myCudaCondDtor(stat, "devPtrB download");

    uchar_to_float(cubeSrc + MAT_SIZE * indRes, tmp);
    stat = cublasSetMatrix(DIM, DIM, sizeof(float), tmp, DIM, devPtrRes, DIM);
    myCudaCondDtor(stat, "devPtrRes download");

    const float alpha = 1;
    const float beta = 1;

    // Single precision real matrices multiplication
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, DIM,
                       DIM, DIM, &alpha, devPtrA, DIM, devPtrB, DIM,
                       &beta, devPtrRes, DIM);
    myCudaCondDtor(stat, "cublasSgemm");

    stat = cublasGetMatrix(DIM, DIM, sizeof(float), devPtrRes, DIM, tmp, DIM);
    // Error codes for gpu to cpu bus
    myCudaCondDtor(stat, "data upload");

    unsigned char *cpuRes = cubeDst + MAT_SIZE * indRes;
    for (int i = 0; i < MAT_SIZE; ++i)
        cpuRes[i] = static_cast<unsigned char>(tmp[i]);
    myCudaFree();
}