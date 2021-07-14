//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
// Experimental tests for aligned memory access and unaligned memory access
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
  REAL *y_cuda, *y, *x;
  REAL a = 123.456;

  n = VEC_LEN;
  fprintf(stderr, "Usage: axpy <n>\n");
  if (argc >= 2) {
    n = atoi(argv[1]);
  }
  y_cuda = (REAL *) malloc(n * sizeof(REAL));
  y  = (REAL *) malloc(n * sizeof(REAL));
  x = (REAL *) malloc(n * sizeof(REAL));

  srand48(1<<12);
  init(x, n);
  init(y_cuda, n);
  memcpy(y, y_cuda, n*sizeof(REAL));

  int i;
  int num_runs = 10;
  for (i=0; i<num_runs; i++) axpy(x, y, n, a);
  /* cuda version */
  double elapsed = read_timer_ms();
  for (i=0; i<num_runs; i++) axpy_cuda(x, y_cuda, n, a);
  elapsed = (read_timer_ms() - elapsed)/num_runs;

  REAL checkresult = check(y_cuda, y, n);
  printf("axpy(%d): checksum: %g, time: %0.2fms\n", n, checkresult, elapsed);
  //assert (checkresult < 1.0e-10);

  free(y_cuda);
  free(y);
  free(x);
  return 0;
}
