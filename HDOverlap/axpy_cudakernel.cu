//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
// Experimental test for new function memcpy_async in CUDA11
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include <cmath>
#include <omp.h>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

/* change this to do saxpy or daxpy : single precision or double precision*/
#define REAL double
#define VEC_LEN 1024000 //use a fixed number for now
/* zero out the entire vector */
void zero(REAL *A, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = 0.0;
    }
}

__global__ 
void
axpy_cudakernel_1perThread(REAL* x, REAL* y, int n, REAL a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > 0 &&i < n) y[i] += a*x[i];
}

double axpy_cuda_normal(REAL* x, REAL* y, int n, REAL a) {
  REAL *d_x, *d_y;
  cudaMalloc(&d_x, n*sizeof(REAL));
  cudaMalloc(&d_y, n*sizeof(REAL));
  double time = read_timer_ms();

  cudaMemcpy(d_x, x, n*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n*sizeof(REAL), cudaMemcpyHostToDevice);
  time = read_timer_ms() - time;

  // Perform axpy elements
  axpy_cudakernel_1perThread<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  cudaDeviceSynchronize();
  
  cudaMemcpy(y, d_y, n*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);
  return time;
}

double axpy_cuda_async(REAL* x, REAL* y, int n, REAL a) {
		cudaStream_t stream1;
		cudaError_t result;
		result = cudaStreamCreate(&stream1);

  REAL *d_x, *d_y;
  cudaMalloc(&d_x, n*sizeof(REAL));
  cudaMalloc(&d_y, n*sizeof(REAL));
  double time2 = read_timer_ms();

  cudaMemcpyAsync(d_x, x, n*sizeof(REAL), cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_y, y, n*sizeof(REAL), cudaMemcpyHostToDevice, stream1);
  time2 = read_timer_ms() - time2;


  //cudaMemcpy(d_x, x, n*sizeof(REAL), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_y, y, n*sizeof(REAL), cudaMemcpyHostToDevice);
    // Perform axpy elements
  axpy_cudakernel_1perThread<<<(n+255)/256, 256>>>(d_x, d_y, n, a);
  cudaDeviceSynchronize();
  

  cudaMemcpy(y, d_y, n*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);
return time2;
}



/* initialize a vector with random floating point numbers */
void init(REAL *A, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = (double)drand48();
    }
}

/*serial version */
void axpy(REAL* x, REAL* y, long n, REAL a) {
  int i;
  for (i = 1; i < n; ++i)
  {
    y[i] += a * x[i];
  }
}

/* compare two arrays and return percentage of difference */
REAL check(REAL*A, REAL*B, int n)
{
    int i;
    REAL diffsum =0.0, sum = 0.0;
    for (i = 0; i < n; i++) {
        diffsum += fabs(A[i] - B[i]);
        sum += fabs(B[i]);
    }
    return diffsum/sum;
}

int main(int argc, char *argv[])
{
  int n;
  REAL *y_cuda, *y, *x, *y_cuda_async;
  REAL a = 123.456;

  n = VEC_LEN;
  fprintf(stderr, "Usage: axpy <n>\n");
  if (argc >= 2) {
    n = atoi(argv[1]);
  }
  y_cuda = (REAL *) malloc(n * sizeof(REAL));
  y_cuda_async = (REAL *) malloc(n * sizeof(REAL));
  y  = (REAL *) malloc(n * sizeof(REAL));
  x = (REAL *) malloc(n * sizeof(REAL));

  srand48(1<<12);
  init(x, n);
  init(y_cuda, n);
  memcpy(y, y_cuda, n*sizeof(REAL));
  memcpy(y_cuda_async, y_cuda, n*sizeof(REAL));

  int i;
  int num_runs = 10;
  for (i=0; i<num_runs; i++) axpy(x, y, n, a);

  //warming up
  axpy_cuda_normal(x, y_cuda_async, n, a);
  axpy_cuda_async(x, y_cuda_async, n, a);

  /* cuda version */
  double elapsed;// = read_timer_ms();
  for (i=0; i<num_runs; i++) elapsed += axpy_cuda_normal(x, y_cuda_async, n, a);
  elapsed =  elapsed/num_runs;

  double elapsed1;// = read_timer_ms();
  for (i=0; i<num_runs; i++) elapsed1 += axpy_cuda_async(x, y_cuda_async, n, a);
  elapsed1 = elapsed1/num_runs;

  REAL checkresult = check(y_cuda, y, n);
  REAL checkresult1 = check(y_cuda_async, y, n);

  printf("axpy(%d): checksum: %g, time: %0.2fms\n", n, checkresult, elapsed);
  printf("axpy_async(%d): checksum: %g, time: %0.2fms\n", n, checkresult1, elapsed1);

  //assert (checkresult < 1.0e-10);

  free(y_cuda);
  free(y);
  free(x);
  return 0;
}
