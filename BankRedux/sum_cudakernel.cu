//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#include "sum.h"

__global__ void sum_warmingup(const REAL *x, REAL *result) {
  __shared__ REAL cache[ThreadsPerBlock];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;
  cache[cacheIndex] = x[tid];
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (cacheIndex < i) {
      cache[cacheIndex] += cache[cacheIndex + i];
    }
    __syncthreads();
  }
  if (cacheIndex == 0)
  result[blockIdx.x] = cache[cacheIndex];
}

__global__ void sum_cudakernel(const REAL *x, REAL *result) {
  __shared__ REAL cache[ThreadsPerBlock];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;
  cache[cacheIndex] = x[tid];
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (cacheIndex < i) {
      cache[cacheIndex] += cache[cacheIndex + i];
    }
    __syncthreads();
  }
  if (cacheIndex == 0)
  result[blockIdx.x] = cache[cacheIndex];
}

__global__ void sum_cudakernel_bc(const REAL *x, REAL *result) {
  __shared__ REAL cache[ThreadsPerBlock];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;
  cache[cacheIndex] = x[tid];
  __syncthreads();
  for (int i = 1; i < blockDim.x; i *= 2) {
    int index = 2 * i * cacheIndex;
    if (index < blockDim.x) {
      cache[index] += cache[index + i];
    }
    __syncthreads();
  }
  if (cacheIndex == 0)
  result[blockIdx.x] = cache[cacheIndex];
}

void sum_cuda(int n, REAL *x, REAL *result) {
  REAL *d_x;
  REAL *d_result;
  cudaMalloc(&d_x, n*sizeof(REAL));
  cudaMalloc(&d_result, ((n+255)/256) * sizeof(REAL));

  cudaMemcpy(d_x, x, n*sizeof(REAL), cudaMemcpyHostToDevice);

  sum_warmingup<<<(n+255)/256, 256>>>(d_x, d_result);
  cudaDeviceSynchronize();
  sum_cudakernel<<<(n+255)/256, 256>>>(d_x, d_result);
  cudaDeviceSynchronize();
  sum_cudakernel_bc<<<(n+255)/256, 256>>>(d_x, d_result);
  cudaDeviceSynchronize();

  cudaMemcpy(result, d_result, ((n+255)/256) * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_result);
}

