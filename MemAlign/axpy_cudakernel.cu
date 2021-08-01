//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#include "axpy.h"

__global__ 
void
axpy_cudakernel_1perThread(REAL* x, REAL* y, int n, REAL a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > 0 &&i < n) y[i] += a*x[i];
}

__global__ 
void
axpy_cudakernel_1perThread_misaligned(REAL* x, REAL* y, int n, REAL a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    if (i < n) y[i] += a*x[i];
}

__global__ 
void
axpy_cudakernel_1perThread_warmup(REAL* x, REAL* y, int n, REAL a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > 1 && i < n) y[i] += a*x[i];
}


void axpy_cuda(REAL* x, REAL* y, int n, REAL a) {
  REAL *d_x, *d_y;
  cudaMalloc(&d_x, n*sizeof(REAL));
  cudaMalloc(&d_y, n*sizeof(REAL));

  cudaMemcpy(d_x, x, n*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n*sizeof(REAL), cudaMemcpyHostToDevice);
  
  //warm up
  axpy_cudakernel_1perThread_warmup<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  cudaDeviceSynchronize();
  // Perform axpy elements
  axpy_cudakernel_1perThread_misaligned<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  cudaDeviceSynchronize();
  axpy_cudakernel_1perThread<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  cudaDeviceSynchronize();
  

  cudaMemcpy(y, d_y, n*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);
}

