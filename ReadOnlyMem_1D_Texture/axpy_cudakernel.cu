//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#include "axpy.h"
#include"cuda_runtime.h"  
#include"device_launch_parameters.h"  
#include <stdio.h>


texture<float, 1, cudaReadModeElementType> rT1;  

__global__ 
void
axpy_cudakernel_warmingup(REAL* x, REAL* y, int n, REAL a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a*x[i];
}


__global__ 
void
axpy_cudakernel_1perThread_texture(REAL* y, int n, REAL a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a * tex1Dfetch(rT1, i);
}

__global__ 
void
axpy_cudakernel_1perThread(REAL* x, REAL* y, int n, REAL a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a*x[i];
}

void axpy_cuda(REAL* x, REAL* y, int n, REAL a) {
  REAL *d_x, *d_y;
  cudaMalloc(&d_x, n*sizeof(REAL));
  cudaMalloc(&d_y, n*sizeof(REAL));

  cudaMemcpy(d_x, x, n*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n*sizeof(REAL), cudaMemcpyHostToDevice);

  cudaBindTexture(0, rT1, d_x);  

  // Perform axpy elements
  axpy_cudakernel_warmingup<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  cudaDeviceSynchronize();
  axpy_cudakernel_1perThread_texture<<<(n+255)/256, 256>>>(d_y, n, a);
  cudaDeviceSynchronize();
  axpy_cudakernel_1perThread<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  cudaDeviceSynchronize();

  cudaMemcpy(y, d_y, n*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaUnbindTexture(rT1);

  cudaFree(d_x);
  cudaFree(d_y);
}
