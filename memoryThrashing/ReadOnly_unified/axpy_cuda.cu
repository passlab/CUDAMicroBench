// Experimental test input for Accelerator directives
//  simplest scalar*vector operations
// Liao 1/15/2013
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "axpy.h"

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
  for (i = 0; i < n; ++i)
  {
    y[i] += a*x[i];
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
    return diffsum;
}

int main(int argc, char *argv[])
{
  int n;
  REAL *x, *y, *y_cuda, *y_cuda1;
  REAL a = 123.456;

  n = VEC_LEN;
  fprintf(stderr, "Usage: axpy <n>\n");
  if (argc >= 2) {
    n = atoi(argv[1]);
  }

  x = (REAL *) malloc(n * sizeof(REAL));
  y  = (REAL *) malloc(n * sizeof(REAL));
  y_cuda = (REAL *) malloc(n * sizeof(REAL));
  y_cuda1 = (REAL *) malloc(n * sizeof(REAL));

  srand48(1<<12);
  init(x, n);
  init(y, n);
  memcpy(y_cuda, y, n*sizeof(REAL));
  memcpy(y_cuda1, y, n*sizeof(REAL));
  
  double elapsed2 = read_timer_ms();
  REAL *x2, *y2;
  cudaMallocManaged(&x2, n*sizeof(REAL));
  cudaMallocManaged(&y2, n*sizeof(REAL));
  elapsed2 = (read_timer_ms() - elapsed2);

  memcpy(y2, y, n*sizeof(REAL));
  memcpy(x2, x, n*sizeof(REAL));


  int i;
  int num_runs = 10;
  
  //serial version
  for (i=0; i<num_runs+1; i++) axpy(x, y, n, a);
  
  
  /* cuda version */
  
  //warming-up
  axpy_cuda(x, y_cuda, n, a);
  
  double elapsed = read_timer_ms();
  for (i=0; i<num_runs; i++) axpy_cuda(x, y_cuda, n, a);
  elapsed = (read_timer_ms() - elapsed)/num_runs;
  
  //using unified memory
  //warming up
  axpy_cudakernel_part1<<<(n+255)/256, 256>>>(x2, y2, n, a);
  cudaDeviceSynchronize();

  axpy_cpu_part2(x2, y2, n, a);

  axpy_cudakernel_part3<<<(n+255)/256, 256>>>(x2, y2, n, a);
  cudaDeviceSynchronize();

  axpy_cpu_part4(x2, y2, n, a);
  cudaDeviceSynchronize();

    
  double elapsed3 = read_timer_ms();
  for (i=0; i<num_runs; i++){
    axpy_cudakernel_part1<<<(n+255)/256, 256>>>(x2, y2, n, a);
    cudaDeviceSynchronize();

    axpy_cpu_part2(x2, y2, n, a);

    axpy_cudakernel_part3<<<(n+255)/256, 256>>>(x2, y2, n, a);
    cudaDeviceSynchronize();

    axpy_cpu_part4(x2, y2, n, a);
  }
  elapsed3 = (read_timer_ms() - elapsed3)/(num_runs);

  
  //warming-up
  axpy_cuda_optimized(x, y_cuda1, n, a);
  
  double elapsed1 = read_timer_ms();
  for (i=0; i<num_runs; i++) axpy_cuda_optimized(x, y_cuda1, n, a);
  elapsed1 = (read_timer_ms() - elapsed1)/num_runs;

  REAL checkresulty = check(y_cuda, y, n);
  REAL checkresulty1 = check(y_cuda1, y, n);
  REAL checkresulty2 = check(y2, y, n);

  printf("axpy(%d): check_y: %g, time: %0.2fms\n", n, checkresulty, elapsed);
  printf("axpy_unified(%d): check_y: %g, time: %0.2fms\n", n, checkresulty2, (elapsed2+elapsed3));
  printf("axpy_optimized(%d): check_y: %g, time: %0.2fms\n", n, checkresulty1, elapsed1);
  //assert (checkresult < 1.0e-10);

  free(y_cuda);
  free(y_cuda1);
  free(y);
  free(x);
  cudaFree(x2);
  cudaFree(y2);

  return 0;
}
