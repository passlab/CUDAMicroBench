#include "axpy.h"

__global__ 
void
axpy_cudakernel(REAL* x, REAL* y, int n, REAL a) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n){
      x[i] = a*x[i];
      y[i] += x[i];
  }
}

__global__ 
void
axpy_cudakernel_part1(REAL* x, REAL* y, int n, REAL a) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < (n/4)){
      x[i] = a*x[i];
      y[i] += x[i];
  }
}

void axpy_cpu_part2(REAL* x, REAL* y, int n, REAL a) {
  for(int i = (n/4); i< (2*(n/4)); i++) {
      x[i] = a*x[i];
      y[i] += x[i];
  } 
}

__global__ 
void
axpy_cudakernel_part3(REAL* x, REAL* y, int n, REAL a) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i>=(2*(n/4))&& i < (3*(n/4))){
      x[i] = a*x[i];
      y[i] += x[i];
  }
}

void axpy_cpu_part4(REAL* x, REAL* y, int n, REAL a) {
  for(int i = (3*(n/4)); i< n; i++) {
      x[i] = a*x[i];
      y[i] += x[i];
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
  REAL *d_x1, *d_y1;
  cudaMalloc(&d_x1, (n/4)*sizeof(REAL));
  cudaMalloc(&d_y1, (n/4)*sizeof(REAL));

  REAL *d_x2, *d_y2;
  cudaMalloc(&d_x2, (n/4)*sizeof(REAL));
  cudaMalloc(&d_y2, (n/4)*sizeof(REAL));

  cudaMemcpy(d_x1, &x[0], (n/4)*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y1, &y[0], (n/4)*sizeof(REAL), cudaMemcpyHostToDevice);

  // Perform axpy elements
  axpy_cudakernel<<<(n+255)/256, 256>>>(d_x1, d_y1, (n/4), a);
  
  cudaMemcpy(&x[0], d_x1, (n/4)*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(&y[0], d_y1, (n/4)*sizeof(REAL), cudaMemcpyDeviceToHost);
  
  axpy_cpu_part2(x, y, n, a);

  cudaMemcpy(d_x2, &x[(2*(n/4))], (n/4)*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y2, &y[(2*(n/4))], (n/4)*sizeof(REAL), cudaMemcpyHostToDevice);

  // Perform axpy elements
  axpy_cudakernel<<<(n+255)/256, 256>>>(d_x2, d_y2, (n/4), a);
  
  cudaMemcpy(&x[(2*(n/4))], d_x2, (n/4)*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(&y[(2*(n/4))], d_y2, (n/4)*sizeof(REAL), cudaMemcpyDeviceToHost);

  axpy_cpu_part4(x, y, n, a);

  cudaFree(d_x1);
  cudaFree(d_y1);
  cudaFree(d_x2);
  cudaFree(d_y2);

}


