// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#include "cudamultiply.cuh"
#include <cublas_v2.h>
#include <iostream>
#include <string>

#define M 4

#define IDX2C(i, j, ld) (((j)*(ld))+(i))
#define IDX(matrix, row, column) ((M*M)*(matrix)+(IDX2C((row),(column),M)))

using std::cerr;

cublasHandle_t handle;
float *devPtrA, *devPtrB, *devPtrRes;

void cuda_free() {
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrRes);
    cublasDestroy(handle);
}

void cuda_free_exit() {
    cuda_free();
    exit(EXIT_FAILURE);
}


void cuda_cond_dtor(cublasStatus_t stat, const char *msg) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << msg << " failed\n";
        cuda_free_exit();
    }
}

// This is so sad. Can we please get integer MMU to CUDA ??
// TODO: change algo so it will reinterpret char as float to avoid conversion ??
void uchar_to_float(const unsigned char *src, float *dst) {
    for (int i = 0; i < M * M; ++i)
        dst[i] = static_cast<float>(src[i]);
}

// Multiply the ind0 matrix with the ind1 matrix, add to the ind_res matrix
// assume all indices are different and matrices are initialized
void
cuda_multiply_add(const unsigned char *cube_src, unsigned char *cube_dst, int
ind0, int ind1, int ind_res) {
    // Error codes for alloc on gpu
    cudaError_t cudaStat0 = cudaMalloc((float **) &devPtrA,
                                       M * M * sizeof(float));
    cudaError_t cudaStat1 = cudaMalloc((float **) &devPtrB,
                                       M * M * sizeof(float));
    cudaError_t cudaStat2 = cudaMalloc((float **) &devPtrRes,
                                       M * M * sizeof(float));
    if (cudaStat0 != cudaSuccess || cudaStat1 != cudaSuccess ||
        cudaStat2 != cudaSuccess) {
        cerr << "device memory allocation" << " failed\n";
        cuda_free_exit();
    }
    cublasStatus_t stat = cublasCreate(&handle);
    // Error codes for CUBLAS initialization
    cuda_cond_dtor(stat, "CuBLAS initialization");



    // Download from cpu to gpu

    float tmp[M * M];
    uchar_to_float(cube_src + M * M * ind0, tmp);
    stat = cublasSetMatrix(M, M, sizeof(float), tmp, M, devPtrA, M);
    // Error codes for cpu to gpu bus
    cuda_cond_dtor(stat, "devPtrA download");

    uchar_to_float(cube_src + M * M * ind1, tmp);
    stat = cublasSetMatrix(M, M, sizeof(float), tmp, M, devPtrB, M);
    cuda_cond_dtor(stat, "devPtrB download");


    const float alpha = 1;
    const float beta = 1;

    // Single precision real matrices multiplication
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M,
                       M, M, &alpha, devPtrA, M, devPtrB, M,
                       &beta, devPtrRes, M);
    cuda_cond_dtor(stat, "cublasSgemm");

    stat = cublasGetMatrix(M, M, sizeof(float), devPtrRes, M, tmp, M);
    // Error codes for gpu to cpu bus
    cuda_cond_dtor(stat, "data upload");

    unsigned char *cpu_res = cube_dst + M * M * ind_res;
    for (int i = 0; i < M * M; ++i)
        cpu_res[i] = static_cast<unsigned char>(tmp[i]);
    cuda_free();
}