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

/*serial version */
void axpy(REAL* x, REAL* y, long n, REAL a, int stride) {
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
  init(y_cuda, (n/stride));
  memcpy(y, y_cuda, (n/stride)*sizeof(REAL));
    memcpy(y, y_cuda_unified, (n/stride)*sizeof(REAL));

  axpy(x, y, n, a, stride);

  int i;
  int num_runs = 10;
  /* cuda version */
  //warming up
  LowAccessDensityTest_cuda(x, y_cuda, n, a, stride);
  
  double elapsed = read_timer_ms();
  for (i=0; i<num_runs; i++) LowAccessDensityTest_cuda(x, y_cuda, n, a, stride);
  elapsed = (read_timer_ms() - elapsed)/num_runs;
  
  //warming up
  LowAccessDensityTest_cuda_unified(x, y_cuda_unified, n, a, stride);
  
  double elapsed1 = read_timer_ms();
  for (i=0; i<num_runs; i++) LowAccessDensityTest_cuda_unified(x, y_cuda_unified, n, a, stride);
  elapsed1 = (read_timer_ms() - elapsed1)/num_runs;
  

  REAL checkresult = check(y_cuda, y, (n/stride));
  REAL checkresult1 = check(y_cuda_unified, y, (n/stride));
  printf("Low Access Test(%ld): check: %g, time: %0.2fms\n", n, checkresult, elapsed);
  printf("Low Access Test(%ld): check_unified: %g, time: %0.2fms\n", n, checkresult1, elapsed1);

  free(y_cuda);
  free(y_cuda_unified);
  free(y);
  free(x);
  return 0;
}
