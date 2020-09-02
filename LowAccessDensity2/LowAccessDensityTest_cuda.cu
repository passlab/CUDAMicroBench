// Experimental test input for Accelerator directives
//  simplest scalar*vector operations
// Liao 1/15/2013
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "LowAccessDensityTest.h"


double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

/* change this to do saxpy or daxpy : single precision or double precision*/
#define REAL float
#define VEC_LEN 102400000//use a fixed number for now
#define STRIDE 1024

/* zero out the entire vector */
void zero(REAL *A, long int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = 0.0;
    }
}

/* initialize a vector with random floating point numbers */
void init(REAL *A, long int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = (double)drand48();
    }
}

__global__ 
void
LowAccessDensityTest_cudakernel(REAL* x, REAL* y, int n, REAL a, int stride)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < (n/stride)) y[i] = a*x[i*stride];
}

//We need to test discrete memory and unified memory, and also for fixed array size (vary stride) 
//and for fix amount of memory access (vary stride)
//So there are total 4 tests. They can use the same CUDA kernel, but the driver functions might need to 
//be separate so it is easier to read.


void LowAccessDensityTest_cuda_discrete_memory(REAL* x, REAL* y, long int n, REAL a, int stride) {
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

/* return the measured time */
double LowAccessDensityTest_cuda_unified_memory(REAL* x, REAL* y, long int n, REAL a, int stride) {
      
  double elapsed1 = read_timer_ms();
  REAL *x2;
  cudaMallocManaged(&x2, n*sizeof(REAL));
  elapsed1 = (read_timer_ms() - elapsed1);

  //initial unified memory, should not count time here
  memcpy(x2, x, n*sizeof(REAL));

  double elapsed2 = read_timer_ms();
  REAL *d_y;
  cudaMalloc(&d_y, (n/stride)*sizeof(REAL));

  LowAccessDensityTest_cudakernel<<<(n+255)/256, 256>>>(x2, d_y, n, a, stride);
  cudaMemcpy(y_cuda_unified, d_y, (n/stride)*sizeof(REAL), cudaMemcpyDeviceToHost);
    
    /* clean up and deallocation */

  elapsed2 = (read_timer_ms() - elapsed2);
    
  return elapsed1 + elapsed2;
}


/*serial version */
void serial(REAL* x, REAL* y, long n, REAL a, int stride) {
  int i;
  for (i = 0; i < (n/stride); i++)
  {
    y[i] = a * x[i*stride];
  }
}

/* compare two arrays and return percentage of difference */
REAL check(REAL*A, REAL*B, long int n)
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
  long int n;
  int stride = STRIDE;
  REAL *y_cuda, *y, *x, *y_cuda_unified;
  REAL a = 123.456;

  n = VEC_LEN;
  fprintf(stderr, "Usage: Low Access Test <n>\n");
  if (argc >= 2) {
    stride = atoi(argv[1]);
  }
  if (argc >= 3) {
    n = atoi(argv[2]);
  }
  y_cuda = (REAL *) malloc((n/stride) * sizeof(REAL));
  y_cuda_unified = (REAL *) malloc((n/stride) * sizeof(REAL));
  y  = (REAL *) malloc((n/stride) * sizeof(REAL));
  x = (REAL *) malloc(n * sizeof(REAL));

  srand48(1<<12);
  init(x, n);

  serial(x, y, n, a, stride);

  int i;
  int num_runs = 10;
  /* cuda version */
  //warming up
  LowAccessDensityTest_cuda(x, y_cuda, n, a, stride);
  
  double elapsed = read_timer_ms();
  for (i=0; i<num_runs;; i++) LowAccessDensityTest_cuda(x, y_cuda, n, a, stride);
  elapsed = (read_timer_ms() - elapsed)/num_runs;
  
  
  double elapsed_unified = 0;
  for (i=0; i<num_runs; i++) elapsed_unified += LowAccessDensityTest_cuda_unified_memory( ... );
  elapsed_unified /= num_runs;

  REAL checkresult = check(y_cuda, y, (n/stride));
  REAL checkresult1 = check(y_cuda_unified, y, (n/stride));
  printf("Low Access Test (Discrete Memory) (%ld): check: %g, time: %0.2fms\n", n, checkresult, elapsed);
  printf("Low Access Test (Unified Memory) (%ld): check_unified: %g, time: %0.2fms\n", n, checkresult1, elapsed_unified);

  free(y_cuda);
  free(y_cuda_unified);
  cudaFree(x2);
  free(y);
  free(x);
  return 0;
}
