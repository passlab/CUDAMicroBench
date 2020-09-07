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

