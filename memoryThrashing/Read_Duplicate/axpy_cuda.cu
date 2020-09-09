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

__global__ 
void
axpy_cudakernel(REAL* x, REAL* y, REAL a, int i) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx>=i && idx < i+512){
    y[i] += a*x[i];
  }
}

void axpy_cpu(REAL* x, REAL* y, REAL a, int i) {
  for(int j = i; j< (i+512); j++) {
    y[j] += a*x[j];
  } 
}

  
double axpy_cuda_unified_memory(REAL* x, REAL * y, long int n, REAL a, REAL* y_cuda) {
  REAL *x2, *y2;
  cudaMallocManaged(&y2, n*sizeof(REAL));
  memcpy(y2, y, n*sizeof(REAL));

  double elapsed1 = read_timer_ms();
  cudaMallocManaged(&x2, n*sizeof(REAL));
  elapsed1 = (read_timer_ms() - elapsed1);

  memcpy(x2, x, n*sizeof(REAL));

  double elapsed2 = read_timer_ms();
  
  for(int i = 0; i< n; i++){
      if((i/512)%2 == 0){
          axpy_cudakernel<<<(n+255)/256, 256>>>(x2, y2, a, i);
                 } else {
       axpy_cpu(x2, y2, a,i);
       }
    }
  cudaDeviceSynchronize();
  elapsed2 = read_timer_ms() - elapsed2;

  memcpy(y_cuda, y2, n*sizeof(REAL));

  cudaFree(x2);
  cudaFree(y2);
  return elapsed1 + elapsed2;
}

double axpy_cuda_unified_memory_optimized(REAL* x, REAL * y, long int n, REAL a, REAL* y_cuda) {
  REAL *x2, *y2;
  cudaMallocManaged(&y2, n*sizeof(REAL));
  memcpy(y2, y, n*sizeof(REAL));

  double elapsed1 = read_timer_ms();
  cudaMallocManaged(&x2, n*sizeof(REAL));
  cudaMemAdvise(x2, n, cudaMemAdviseSetReadMostly, 0);
  cudaMemPrefetchAsync(x2, n, 0);
  elapsed1 = (read_timer_ms() - elapsed1);

  memcpy(x2, x, n*sizeof(REAL));

  double elapsed2 = read_timer_ms();
  
  for(int i = 0; i< n; i++){
      if((i/512)%2 == 0){
          axpy_cudakernel<<<(n+255)/256, 256>>>(x2, y2, a, i);
      } else {
         axpy_cpu(x2, y2, a,i);
       }
    }
  cudaDeviceSynchronize();

  elapsed2 = read_timer_ms() - elapsed2;

  memcpy(y_cuda, y2, n*sizeof(REAL));

  cudaFree(x2);
  cudaFree(y2);
  return elapsed1 + elapsed2;
}

int main(int argc, char *argv[])
{
  int n;
  REAL *x, *y, *y_cuda, *y_cuda1, *y_serial;
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
  y_serial  = (REAL *) malloc(n * sizeof(REAL));


  srand48(1<<12);
  init(x, n);
  zero(y, n);
  memcpy(y_serial, y, n*sizeof(REAL));
  memcpy(y_cuda, y, n*sizeof(REAL));
  memcpy(y_cuda1, y, n*sizeof(REAL));

  int i;
  int num_runs = 50;
  
  //serial version
  axpy(x, y_serial, n, a);
  
  //warm up
  double warm = axpy_cuda_unified_memory(x, y, n, a, y_cuda);



  double elapsed_unified = 0;
  for (i=0; i<num_runs; i++) elapsed_unified += axpy_cuda_unified_memory(x, y, n, a, y_cuda);
  elapsed_unified /= num_runs;

  double elapsed_unified_optimized = 0;
  for (i=0; i<num_runs; i++) elapsed_unified_optimized += axpy_cuda_unified_memory_optimized(x, y, n, a, y_cuda1);
  elapsed_unified_optimized /= num_runs;


  REAL checkresulty = check(y_cuda, y_serial, n);
  REAL checkresulty1 = check(y_cuda1, y_serial, n);

  //printf("axpy(%d): time: %0.2fms\n", n, elapsed);
  printf("axpy_unified(%d): check_y: %g, time: %0.2fms\n", n, checkresulty, elapsed_unified);
  printf("axpy_optimized(%d): check_y: %g, time: %0.2fms\n", n, checkresulty1, elapsed_unified_optimized);
  //assert (checkresult < 1.0e-10);

  free(y);
  free(x);
  return 0;
}