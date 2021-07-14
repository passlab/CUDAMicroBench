//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#include "warpDivergenceTest.h"


__global__ void warmingup(float *x, float *y, float *z) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid % 2 == 0) {
        z[tid] = 2 * x[tid] + 3 * y[tid];

    } else {
        z[tid] = 3 * x[tid] + 2 * y[tid];
    }
}

__global__ void warpDivergence(float *x, float *y, float *z) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid % 2 == 0) {
        z[tid] = 2 * x[tid] + 3 * y[tid];

    } else {
        z[tid] = 3 * x[tid] + 2 * y[tid];
    }
}

__global__ void noWarpDivergence(float *x, float *y, float *z) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((tid / warpSize) % 2 == 0) {
        z[tid] = 2 * x[tid] + 3 * y[tid];
    } else {
        z[tid] = 3 * x[tid] + 2 * y[tid];
    }
}
void warpDivergenceTest_cuda(REAL* x, REAL* y, REAL *warp_divergence, REAL *no_warp_divergence, int n) {
  REAL *d_x, *d_y, *d_warp_divergence, *d_no_warp_divergence;
  cudaMalloc(&d_x, n*sizeof(REAL));
  cudaMalloc(&d_y, n*sizeof(REAL));
  cudaMalloc(&d_warp_divergence, n*sizeof(REAL));
  cudaMalloc(&d_no_warp_divergence, n*sizeof(REAL));

  cudaMemcpy(d_x, x, n*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n*sizeof(REAL), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  warmingup<<<(n+255)/256, 256>>> (d_x, d_y, d_warp_divergence);
  cudaDeviceSynchronize();

  warpDivergence<<<(n+255)/256, 256>>>(d_x, d_y, d_warp_divergence);
  cudaDeviceSynchronize();

  noWarpDivergence<<<(n+255)/256, 256>>>(d_x, d_y, d_no_warp_divergence);
  cudaDeviceSynchronize();

  cudaMemcpy(warp_divergence, d_warp_divergence, n*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(no_warp_divergence, d_no_warp_divergence, n*sizeof(REAL), cudaMemcpyDeviceToHost);


  cudaFree(d_x);
  cudaFree(d_y);

  cudaFree(d_warp_divergence);
  cudaFree(d_no_warp_divergence);


}
