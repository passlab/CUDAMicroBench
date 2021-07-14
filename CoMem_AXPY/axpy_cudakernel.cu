//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#include "axpy.h"

__global__ 
void
axpy_cudakernel_warmingup(REAL* x, REAL* y, int n, REAL a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a*x[i];
}

__global__ 
void
axpy_cudakernel_1perThread(REAL* x, REAL* y, int n, REAL a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a*x[i];
}

/* block distribution of loop iteration */
__global__ 
void axpy_cudakernel_block(REAL* x, REAL* y, int n, REAL a) {
	int thread_num = threadIdx.x + blockIdx.x * blockDim.x;
	int total_threads = gridDim.x * blockDim.x;

	int block_size = n / total_threads; //dividable, TODO handle non-dividiable later
	
	int start_index = thread_num * block_size;
	int stop_index = start_index + block_size;
	int i;
        for (i=start_index; i<stop_index; i++) {
		if (i < n) y[i] += a*x[i];
	}
}

/* cyclic distribution of loop distribution */
__global__
void axpy_cudakernel_cyclic(REAL* x, REAL* y, int n, REAL a) {
	int thread_num = threadIdx.x + blockIdx.x * blockDim.x;
	int total_threads = gridDim.x * blockDim.x;
	
	int i;
	for (i=thread_num; i<n; i+=total_threads) { 
		if (i < n) y[i] += a*x[i];
	}
}

void axpy_cuda(REAL* x, REAL* y, int n, REAL a) {
  REAL *d_x, *d_y;
  cudaMalloc(&d_x, n*sizeof(REAL));
  cudaMalloc(&d_y, n*sizeof(REAL));

  cudaMemcpy(d_x, x, n*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n*sizeof(REAL), cudaMemcpyHostToDevice);

  // Perform axpy elements
  axpy_cudakernel_warmingup<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  cudaDeviceSynchronize();
  axpy_cudakernel_1perThread<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  cudaDeviceSynchronize();
  axpy_cudakernel_block<<<1024, 256>>>(d_x, d_y, n, a);
  cudaDeviceSynchronize();
  axpy_cudakernel_cyclic<<<1024, 256>>>(d_x, d_y, n, a);
  cudaDeviceSynchronize();

  cudaMemcpy(y, d_y, n*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);
}

