//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
// Experimental test for bank conflict
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "sum.h"

double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

/* change this to do saxpy or daxpy : single precision or double precision*/
#define REAL float

//#define ThreadsPerBlock 256

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
        A[i] = (float)drand48();
    }
}

/*serial version */
float sum(int N, float *numbers) {
	float sum = 0;
	
	for (int i = 0; i<N; i++)
		sum += numbers[i];
	
	return sum;
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
  REAL *x;
  REAL *result_cuda;

  n = VEC_LEN;
  fprintf(stderr, "Usage: sum <n>\n");
  if (argc >= 2) {
    n = atoi(argv[1]);
  }
  
  x = (REAL *) malloc(n * sizeof(REAL));
  result_cuda = (REAL*)malloc(((VEC_LEN + ThreadsPerBlock - 1) / ThreadsPerBlock) * sizeof(REAL));

  srand48(1<<12);
  init(x, n);
  
  volatile float answer = 0;
  answer = sum(n, x);

  int i;
  int num_runs = 10;
  /* cuda version */
  double elapsed = read_timer_ms();
  for (i=0; i<num_runs; i++) sum_cuda(n, x, result_cuda);
  
 	for (int i = 1; i < ((VEC_LEN + ThreadsPerBlock - 1) / ThreadsPerBlock); ++i)
  result_cuda[0] += result_cuda[i];

  elapsed = (read_timer_ms() - elapsed)/num_runs;
  printf("sum(%d): checksum: %g, time: %0.2fms\n", n, result_cuda[0]-answer, elapsed);


  free(x);
  return 0;
}
