//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
// Experimental test for low memory access density using unified memory
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include <time.h>

double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

/* change this to do saxpy or daxpy : single precision or double precision*/
#define REAL double
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
void serial_kernel(REAL* x, REAL* y, long n, REAL a, int stride) {
  int i;
  for (i = 0; i < n; i+=stride)
  {
    y[i] += a * x[i];
  }
}

/*omp version */
void omp_kernel(REAL* x, REAL* y, long n, REAL a, int stride) {
  int i;
  #pragma omp parallel for shared(x,y,a,n,stride) private(i)  
  for (i = 0; i < n; i+=stride)
  {
    y[i] += a * x[i];
  }
}

/*omp gpu version */
void omp_gpu_kernel(REAL* x, REAL* y, long n, REAL a, int stride) {
  int i;
  //#pragma omp target teams distribute parallel for map(tofrom:y) map(to:x,a,n,stride)
  #pragma omp target map(to:a,n,x[0:n]) map(tofrom:y[0:n])
  #pragma parallel for
  for (i = 0; i < n; i+=stride)
  {
    y[i] += a * x[i];
  }
}



/* compare two arrays and return percentage of difference */
REAL check(REAL*A, REAL*B, long int n)
{
    int i;
    REAL diffRatioSum= 0.0;
    for (i = 0; i < n; i++) {
      REAL diff = fabs(A[i] - B[i]);
      if (fabs(B[i])==0.0)
	diffRatioSum+=0.0;
      else
	diffRatioSum += diff/fabs(B[i]);
    }
    return diffRatioSum/n;
}

int main(int argc, char *argv[])
{
  long int n;
  int stride = STRIDE;
  REAL *y_omp, *y, *x;
  REAL a = 123.456;

  n = VEC_LEN;
  fprintf(stderr, "Usage: %s <stride> [vec_len]\n", argv[0]);
  if (argc >= 2) {
    stride = atoi(argv[1]);
  }

  if (argc >= 3) {
    n = atoi(argv[2]);
  }
  printf("vec len(n_=%ld, stride=%d\n", n, stride);

  // same input x
  x = (REAL *) malloc(n * sizeof(REAL));
  if (x==NULL)
  {
    fprintf(stderr, "malloc returns NULL: out of memory\n");
    abort();
  }
  srand48(time(NULL));
  init(x, n);

  // output for serial and omp version
  y  = (REAL *) malloc(n * sizeof(REAL));
  if (y==NULL)
  {
    fprintf(stderr, "y malloc returns NULL: out of memory\n");
    abort();
  }

  y_omp  = (REAL *) malloc(n * sizeof(REAL));
  if (y_omp==NULL)
  {
    fprintf(stderr, "y_omp malloc returns NULL: out of memory\n");
    abort();
  }

  REAL*  y_omp_gpu  = (REAL *) malloc(n * sizeof(REAL));
  if (y_omp_gpu==NULL)
  {
    fprintf(stderr, "y_omp malloc returns NULL: out of memory\n");
    abort();
  }


  // serial version as a reference
  serial_kernel(x, y, n, a, stride);

  int i;
  int num_runs = 100;

  /* OMP version */
  double elapsed = read_timer_ms();
  for (i=0; i<num_runs; i++) 
    omp_kernel(x, y_omp, n, a, stride);
  elapsed = (read_timer_ms() - elapsed)/num_runs;

  printf("diff ratio=%f\n", check(y,y_omp, n));

  for (i=0; i<num_runs; i++) 
    omp_gpu_kernel(x, y_omp_gpu, n, a, stride);

  printf("diff ratio=%f\n", check(y,y_omp_gpu, n));

  free(x);
  free(y);
  free(y_omp);
  free(y_omp_gpu);
  return 0;
}
