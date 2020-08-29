#include "axpy.h"

__global__ 
void
axpy_cudakernel(REAL* x, REAL* y, int n, REAL a) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n){
    y[i] += a*x[i];
  }
}

__global__ 
void
axpy_cudakernel_part1(REAL* x, REAL* y, int n, REAL a) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < (n/4)){
    y[i] += a*x[i];
  }
}

void axpy_cpu_part2(REAL* x, REAL* y, int n, REAL a) {
  for(int i = (n/4); i< (2*(n/4)); i++) {
    y[i] += a*x[i];
  } 
}

__global__ 
void
axpy_cudakernel_part3(REAL* x, REAL* y, int n, REAL a) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i>=(2*(n/4))&& i < (3*(n/4))){
    y[i] += a*x[i];
  }
}

void axpy_cpu_part4(REAL* x, REAL* y, int n, REAL a) {
  for(int i = (3*(n/4)); i< n; i++) {
    y[i] += a*x[i];
  } 
}

void axpy_cuda(REAL* x, REAL* y, int n, REAL a) {
  REAL *d_x, *d_y;
  cudaMalloc(&d_x, n*sizeof(REAL));
  cudaMalloc(&d_y, n*sizeof(REAL));

  cudaMemcpy(d_x, x, n*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n*sizeof(REAL), cudaMemcpyHostToDevice);

  // Perform axpy elements
  axpy_cudakernel_part1<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  
  cudaMemcpy(x, d_x, n*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(y, d_y, n*sizeof(REAL), cudaMemcpyDeviceToHost);
  
  axpy_cpu_part2(x, y, n, a);

  cudaMemcpy(d_x, x, n*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n*sizeof(REAL), cudaMemcpyHostToDevice);

  // Perform axpy elements
  axpy_cudakernel_part3<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  
  cudaMemcpy(x, d_x, n*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(y, d_y, n*sizeof(REAL), cudaMemcpyDeviceToHost);

  axpy_cpu_part4(x, y, n, a);

  
  cudaFree(d_x);
  cudaFree(d_y);
}

void axpy_cuda_optimized(REAL* x, REAL* y, int n, REAL a) {
  REAL *d_x, *d_y;
  cudaMalloc(&d_x, n*sizeof(REAL));
  cudaMalloc(&d_y, n*sizeof(REAL));

  cudaMemcpy(d_x, x, n*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n*sizeof(REAL), cudaMemcpyHostToDevice);

  // Perform axpy elements
  axpy_cudakernel_part1<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  
  cudaMemcpy(y, d_y, n*sizeof(REAL), cudaMemcpyDeviceToHost);
  
  axpy_cpu_part2(x, y, n, a);

  cudaMemcpy(d_y, y, n*sizeof(REAL), cudaMemcpyHostToDevice);

  // Perform axpy elements
  axpy_cudakernel_part3<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  
  cudaMemcpy(y, d_y, n*sizeof(REAL), cudaMemcpyDeviceToHost);

  axpy_cpu_part4(x, y, n, a);

  
  cudaFree(d_x);
  cudaFree(d_y);

}


