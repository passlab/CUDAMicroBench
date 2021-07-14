//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#include "mm_omp_cuda.h"
#include <stdio.h>
#define BLOCK_SIZE 16

__global__
void global_element(REAL* A, REAL* B, REAL* C, int n) {

    REAL C_value = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int k = 0; k < n; k++) {
        C_value += A[row * n + k] * B[n * k + col];
    }

    // Each thread writes one element to C matrix
    C[row * n + col] = C_value;
}

__global__
void global_block(REAL* A, REAL* B, REAL* C, int n) {
    int wA = n;
    int wB = n;

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
      for (int k = 0; k < BLOCK_SIZE; ++k) {
          Csub += A[a + wA * ty + k] * B[b + wB * k + tx];
      }
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__global__
void shared_block(REAL* A, REAL* B, REAL* C, int n) {
    int wA = n;
    int wB = n;

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ REAL As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ REAL Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void mm_kernel(REAL* A, REAL* B, REAL* C, int n) {
    REAL *A_device, *B_device, *C_device;
    cudaMalloc(&A_device, n*n*sizeof(REAL));
    cudaMalloc(&B_device, n*n*sizeof(REAL));
    cudaMalloc(&C_device, n*n*sizeof(REAL));

    cudaMemcpy(A_device, A, n*n*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, n*n*sizeof(REAL), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(n / dimBlock.x, n / dimBlock.y);
    //global_element<<<dimGrid, dimBlock>>>(A_device, B_device, C_device, n);
    global_block<<<dimGrid, dimBlock>>>(A_device, B_device, C_device, n);
    //shared_block<<<dimGrid, dimBlock>>>(A_device, B_device, C_device, n);

    cudaMemcpy(C, C_device, n*n*sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
}

void mm_kernel_shmem(REAL* A, REAL* B, REAL* C, int n) {
    REAL *A_device, *B_device, *C_device;
    cudaMalloc(&A_device, n*n*sizeof(REAL));
    cudaMalloc(&B_device, n*n*sizeof(REAL));
    cudaMalloc(&C_device, n*n*sizeof(REAL));

    cudaMemcpy(A_device, A, n*n*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, n*n*sizeof(REAL), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(n / dimBlock.x, n / dimBlock.y);
    //global_element<<<dimGrid, dimBlock>>>(A_device, B_device, C_device, n);
    //global_block<<<dimGrid, dimBlock>>>(A_device, B_device, C_device, n);
    shared_block<<<dimGrid, dimBlock>>>(A_device, B_device, C_device, n);

    cudaMemcpy(C, C_device, n*n*sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
}

