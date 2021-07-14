//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
// Experimental tests for warp divergence
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "warpDivergenceTest.h"

double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

/* change this to do saxpy or daxpy : single precision or double precision*/
#define REAL float
#define VEC_LEN 32000 //use a fixed number for now
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
void warpDivergenceSerial(REAL* x, REAL* y, REAL* z, int n) {
  int i;
  for (i = 0; i < n; ++i)
  {
    if(i%2 == 0) z[i] = 2 * x[i] + 3 * y[i];
    else z[i] = 3 * x[i] + 2 * y[i];
  }
}

void NoWarpDivergenceSerial(REAL* x, REAL* y, REAL* z, int n) {
  int i;
  for (i = 0; i < n; ++i)
  {
    if((i/32)%2 ==0 ) z[i] = 2 * x[i] + 3 * y[i];
    else z[i] = 3 * x[i] + 2 * y[i];
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
  REAL *x, *y, *warp_divergence, *no_warp_divergence, *warp_divergence_serial, *no_warp_divergence_serial;

  n = VEC_LEN;
  fprintf(stderr, "Usage: warpDivergenceTest <n>\n");
  if (argc >= 2) {
    n = atoi(argv[1]);
  }
  x = (REAL *) malloc(n * sizeof(REAL));
  y = (REAL *) malloc(n * sizeof(REAL));
  warp_divergence = (REAL *) malloc(n * sizeof(REAL));
  no_warp_divergence = (REAL *) malloc(n * sizeof(REAL));
  
  warp_divergence_serial = (REAL *) malloc(n * sizeof(REAL));
  no_warp_divergence_serial = (REAL *) malloc(n * sizeof(REAL));
    

  srand48(1<<12);
  init(x, n);
  //init(y, n);
  memcpy(y, x, n*sizeof(REAL));


  int i;
  int num_runs = 10;
  
  warpDivergenceSerial(x,y,warp_divergence_serial,n);
  NoWarpDivergenceSerial(x,y,no_warp_divergence_serial,n);
  /* cuda version */
  double elapsed = read_timer_ms();
  for (i=0; i<num_runs; i++) warpDivergenceTest_cuda(x, y, warp_divergence, no_warp_divergence, n);
  elapsed = (read_timer_ms() - elapsed)/num_runs;

  float check1 = check(warp_divergence,warp_divergence_serial,n);
  float check2 = check(no_warp_divergence,no_warp_divergence_serial,n);
  printf("check:%f\n", check1);
  printf("check:%f\n", check2);
  //assert (checkresult < 1.0e-10);

  free(x);
  free(y);
  free(warp_divergence);
  free(no_warp_divergence);
  free(warp_divergence_serial);
  free(no_warp_divergence_serial);
  return 0;
}
