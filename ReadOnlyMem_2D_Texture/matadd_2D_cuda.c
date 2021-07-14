//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
// Experimental test for texture memory using 2-D array
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "matadd_2D.h"
#include<cuda_runtime_api.h> 
#include<device_launch_parameters.h> 
#include<stdio.h>
#include<time.h>
 
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define VEC_LEN 1024//use a fixed number for now


/* zero out the entire vector */
void zero(REAL *A, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = 0.0;
    }
}

/* initialize a matrix with random REALing point numbers */
void init_matrix(REAL *matrix, int m, int n) {
	for (int i = 0; i<m; i++) {
		for (int j = 0; j<n; j++) {
			matrix[i*n+j] = 1;//(REAL)drand48();//(REAL)rand()/(REAL)(RAND_MAX/10.0);
		}
	}
}

void printMatrix(REAL *pflMat, int M, int N)
{
	for(int idxM = 0; idxM < M; idxM++)
	{
		for(int idxN = 0; idxN < N; idxN++)
		{
			printf("%f\t",pflMat[(idxM * N) + idxN]);
		}
		printf("\n");
	}
	printf("\n");
}


/*serial version */
void mat_add_serial(REAL* x, REAL* y, int m, int n, REAL* result) {
  int i;
  for (i = 0; i < m; i++) {
    for(int j = 0; j < n; j++){
      result[i*n+j] = x[i*n+j] + y[i*n+j];
    }
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
    int N;
    N = VEC_LEN;
    fprintf(stderr, "Usage: MatAdd <N*M>\n");
    if (argc >= 2) {
      N = atoi(argv[1]);
    }
  
  	int M=N;

  	REAL *h_matrixA = (REAL*)malloc(M * N * sizeof(REAL));
  	REAL *h_matrixB = (REAL*)malloc(M * N * sizeof(REAL));
  	REAL *h_result = (REAL*)malloc(M * N * sizeof(REAL));
   	REAL *result_serial = (REAL*)malloc(M * N * sizeof(REAL));
 
    init_matrix(h_matrixA, M, N);
    init_matrix(h_matrixB, M, N);
 
    int i;
    int num_runs = 5;
    mat_add_serial(h_matrixA, h_matrixB, M, N, result_serial);
    for (i=0; i<num_runs; i++) matadd(h_matrixA, h_matrixB, M, N, h_result);
    printf("check:%f\n", check(result_serial,h_result,M*N));
  
}
