//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
/*
 * Square matrix multiplication
 * A[N][N] * B[N][N] = C[N][N]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>
#include "mm_omp_cuda.h"

#define ALLOWED_DIFF 0.0001

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define REAL double

void init(int N, REAL *A) {
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i*N+j] = (REAL) drand48();
        }
    }
}


void matmul_serial(int N, REAL *A, REAL *B, REAL *C) {
    int i,j,k;
    REAL temp;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            temp = 0;
            for (k = 0; k < N; k++) {
                temp += (A[i * N + k] * B[k * N + j]);
            }
            C[i * N + j] = temp;
        }
    }
}

int main(int argc, char *argv[]) {
    int N;

    int num_threads = 4; /* 4 is default number of threads */
    if (argc < 2) {
        fprintf(stderr, "Usage: mm <n> (default %d) [<num_threads>] (default %d)\n", N, num_threads);
        exit(1);
    }
    N = atoi(argv[1]);

    double elapsed_shmem;
    double elapsed_cuda;

    REAL *A = malloc(sizeof(REAL)*N*N);
    REAL *B = malloc(sizeof(REAL)*N*N);
    REAL *C_shmem = malloc(sizeof(REAL)*N*N);
    REAL *C = malloc(sizeof(REAL)*N*N);
    REAL *C_serial = malloc(sizeof(REAL)*N*N);

    srand48((1 << 12));
    init(N, A);
    init(N, B);

    int i, j;
    int num_runs = 10;
    
    matmul_serial(N, A, B, C_serial);
    mm_kernel_shmem(A, B, C_shmem,N);
    
    elapsed_cuda = read_timer();
    for (i=0; i<num_runs; i++)
        mm_kernel(A, B, C, N);
    elapsed_cuda = (read_timer() - elapsed_cuda)/num_runs;
    
    elapsed_shmem = read_timer();
    for (i=0; i<num_runs; i++)
        mm_kernel_shmem(A, B, C_shmem,N);
    elapsed_shmem = (read_timer() - elapsed_shmem)/num_runs;
    /* you should add the call to each function and time the execution */


    
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (fabs(C[i * N + j] - C_serial[i * N + j]) > ALLOWED_DIFF) {
                printf("C[%d][%d]: %g, C_omp[%d][%d]: %g\n", i, j, C[i * N + j], i, j, C_serial[i * N + j]);
                break;
            }
        }
    };

    printf("======================================================================================================\n");
    printf("\tMatrix Multiplication: A[N][N] * B[N][N] = C[N][N], N=%d\n", N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matmul_cuda:\t\t%4f\t%4f\n", elapsed_cuda * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda)));
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matmul_shmem:\t\t%4f\t%4f\n", elapsed_shmem * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_shmem)));

    return 0;
}


