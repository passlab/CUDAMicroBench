#include "LowAccessDensityTest.h"

__global__ 
void
LowAccessDensityTest_cudakernel(REAL* x, REAL* y, int n, REAL a, int stride)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < (n/stride)) y[i] = a*x[i*stride];
}

__global__ 
void
LowAccessDensityTest_cudakernel_unified(REAL* x, REAL* y, int n, REAL a, int stride)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < (n/stride)) y[i] = a*x[i];
}

void LowAccessDensityTest_cuda(REAL* x, REAL* y, long int n, REAL a, int stride) {
  REAL *d_x, *d_y;
  cudaMalloc(&d_x, n*sizeof(REAL));
  cudaMalloc(&d_y, (n/stride)*sizeof(REAL));

  cudaMemcpy(d_x, x, n*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, (n/stride)*sizeof(REAL), cudaMemcpyHostToDevice);

  LowAccessDensityTest_cudakernel<<<(n+255)/256, 256>>>(d_x, d_y, n, a, stride);

  cudaMemcpy(y, d_y, (n/stride)*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);
}

void LowAccessDensityTest_cuda_unified(REAL* x, REAL* y, long int n, REAL a, int stride) {
  REAL *d_x, *d_y;
  cudaMallocManaged(&d_x, (n/stride)*sizeof(REAL));
  cudaMalloc(&d_y, (n/stride)*sizeof(REAL));

  for(int i = 0; i < (n/stride); i++) {
      d_x[i] = x[i*stride];
  }
  cudaMemcpy(d_y, y, (n/stride)*sizeof(REAL), cudaMemcpyHostToDevice);
  LowAccessDensityTest_cudakernel_unified<<<(n+255)/256, 256>>>(d_x, d_y, n, a, stride);
  cudaMemcpy(y, d_y, (n/stride)*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);

}


