// SPDX-License-Identifier: GPLv3-or-later
// Copyright © 2020 Or Toledano
#include "cudamultiply.cuh"
#include <cublas_v2.h>
#include <cstdio>


#define M 4
#define MAT_MULT_ITERS 64

#define IDX2C(i, j, ld) (((j)*(ld))+(i))
#define IDX(matrix, row, column) ((M*M)*(matrix)+(IDX2C((row),(column),M)))



// Multiply the ind0 matrix with the ind1 matrix, add to the ind_res matrix
// assume all indices are different and matrices are initialized
void
cuda_multiply_add(const unsigned char *cube_src, unsigned char *cube_dst, int ind0,
             int ind1, int ind_res) {
    float *devPtrA, *devPtrB, *devPtrRes;
    // Error codes for alloc on gpu
    cudaError_t cudaStat1 = cudaMalloc((float **) &devPtrA,
                                       M * M * sizeof(float));
    cudaError_t cudaStat2 = cudaMalloc((float **) &devPtrB,
                                       M * M * sizeof(float));
    cudaError_t cudaStat3 = cudaMalloc((float **) &devPtrRes,
                                       M * M * sizeof(float));
    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess ||
        cudaStat3 != cudaSuccess) {
        printf("device memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    cublasHandle_t handle;
    cublasStatus_t stat1 = cublasCreate(&handle);
    // Error codes for CUBLAS initialization
    if (stat1 != CUBLAS_STATUS_SUCCESS) {
        printf("CuBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
    // Download from cpu to gpu
    stat1 = cublasSetMatrix(M, M, sizeof(*cube_src), cube_src + M * M * ind0, M,
                            devPtrA, M);
    cublasStatus_t stat2 = cublasSetMatrix(M, M, sizeof(*cube_src),
                                           cube_src + M * M * ind1, M, devPtrA,
                                           M);
    // Error codes for cpu to gpu bus
    if (stat1 != CUBLAS_STATUS_SUCCESS || stat2 != CUBLAS_STATUS_SUCCESS) {
        printf("data download failed\n");
        cudaFree(devPtrA);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }

    const float alpha = 1;
    const float beta = 1;

    // Single precision real matrices multiplication
    cublasStatus_t statMult = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M,
                                          M, M, &alpha, devPtrA, M, devPtrB, M,
                                          &beta, devPtrRes, M);
    if (statMult != CUBLAS_STATUS_SUCCESS) {
        printf("CuBLAS gemm failed\n");
        exit(EXIT_FAILURE);
    }

    cublasStatus_t statRes = cublasGetMatrix(M, M, sizeof(float), devPtrA, M,
                                             cube_dst + M * M * ind_res, M);
    // Error codes for gpu to cpu bus
    if (statRes != CUBLAS_STATUS_SUCCESS) {
        printf("data upload failed\n");
        cudaFree(devPtrA);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }

    cublasDestroy(handle);
    cudaFree(devPtrRes);
    cudaFree(devPtrB);
    cudaFree(devPtrA);
    exit(EXIT_SUCCESS);
}