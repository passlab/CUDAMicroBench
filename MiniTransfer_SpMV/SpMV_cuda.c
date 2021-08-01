//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
// Experimental test for minimizing data transfer using sparseMV
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "SpMV.h"
#include <time.h>

double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}



void zero(REAL *A, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = 0.0;
    }
}

/* initialize a vector with random floating point numbers */
void init_vector(REAL *vector, int m)
{
	for (int i = 0; i<m; i++) {
    vector[i] = (REAL)drand48();//(float)rand()/(float)(RAND_MAX/10.0);
  }
}



void init_matrix(REAL * matrix, int num_rows,int nnz) {

    REAL *d, *A;
    d = (REAL *) malloc((num_rows * num_rows) * sizeof(REAL));
    A = (REAL *) malloc((num_rows * num_rows) * sizeof(REAL));
    int i,j,n,a,b,t;
    srand(time(NULL));
    n=num_rows*num_rows;
    for (i=0;i<n;i++) d[i]=i;
    for (i=n;i>0;i--) {
        a=i-1;b=rand()%i;
        if (a!=b) {t=d[a];d[a]=d[b];d[b]=t;}
    }
 
    for (i=0;i<num_rows;i++) {
        for (j=0;j<num_rows;j++) {
            A[i*num_rows+j]=(d[i*num_rows+j]>=nnz)?0:(REAL)(drand48()+1);
            matrix[i*num_rows+j] = A[i*num_rows+j];
        }
    }
}

void init_csr(int *ptr, REAL *data, int *indices, REAL *matrix, int num_rows, int nnz)
{
  int tmp = 0;
  ptr[num_rows] = nnz;
  ptr[0] = 0;
  float * non_zero_elements;
  non_zero_elements = (float *) malloc(num_rows * sizeof(float));
	for (int i = 0; i < num_rows; i++) {
    int tmp1 = 0;
	  for (int j = 0; j < num_rows; j++) {
      if(matrix[i*num_rows+j] != 0) {
        data[tmp] = matrix[i*num_rows+j];
        indices[tmp] = j;
        tmp++;
        tmp1++;
      }
	  non_zero_elements[i] = tmp1;
    }
  }
  for (int i = 1; i<num_rows; i++){
    ptr[i] = ptr[i-1]+ non_zero_elements[i-1];
  }
}

void init_index_count(int * row_nnz_start, int * row, int * column, REAL *matrix, int num_rows)
{
  int tmp = 0;
  int count = 0;
  row_nnz_start[0] = 0;
	for (int i = 0; i < num_rows; i++) {
    count = 0;
	  for (int j = 0; j < num_rows; j++) {
      if(matrix[i*num_rows+j] != 0) {
        count++;
        row[tmp] = i;
        column[tmp] = j;
        tmp++;
      }
    }
    if( i < num_rows-1 ) row_nnz_start[i+1] = row_nnz_start[i] + count;
  }
}

void init_index(int * row, int * column, REAL *matrix, int num_rows)
{
  int tmp = 0;
	for (int i = 0; i < num_rows; i++) {
	  for (int j = 0; j < num_rows; j++) {
      if(matrix[i*num_rows+j] != 0) {
        row[tmp] = i;
        column[tmp] = j;
        tmp++;
      }
    }
  }
}

void spmv_csr_serial( const int num_rows, const int *ptr, const int * indices, const float *data, const float * x, float *y){
    for(int row = 0; row < num_rows; row++){
        float dot = 0;
        int row_start = ptr[row];
        int row_end = ptr[row+1];
        
        for ( int i = row_start; i < row_end; i++) dot += data[i] * x[indices[i]];
        y[row] += dot;
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
  int *ptr, * indices;
  float * data, * x, *y, *y_csr, *y_dense, *y_warmingup, *y_unified, *y_unified_count;
  float *matrix;
  int num_rows = 0;
  int nnz = 0;
  //fprintf(stderr, "Usage: SpMV <n>\n");
  if (argc >= 2) {
    nnz = atoi(argv[1]);
  }
  if (argc >= 3) {
    num_rows = atoi(argv[2]);
  }
  
  ptr = (int *) malloc((num_rows+1) * sizeof(int));
  indices = (int *) malloc(nnz * sizeof(int));
  data = (float *) malloc(nnz * sizeof(float));
  x = (float *) malloc(num_rows * sizeof(float));
  y = (float *) malloc(num_rows * sizeof(float));
  y_csr = (float *) malloc(num_rows * sizeof(float));
  y_dense = (float *) malloc(num_rows * sizeof(float));
  y_unified = (float *) malloc(num_rows * sizeof(float));
  y_unified_count = (float *) malloc(num_rows * sizeof(float));
  y_warmingup = (float *) malloc(num_rows * sizeof(float));
  matrix = (float *) malloc(num_rows * num_rows * sizeof(float));

  srand48(1<<12);
  init_matrix(matrix,num_rows,nnz);

  init_vector(x, num_rows);
  init_csr(ptr, data,indices, matrix, num_rows,nnz);
  //init_ptr(ptr, matrix,num_rows,nnz);

  int i;
  int num_runs = 5;
  /* cuda version */
  //double elapsed = read_timer_ms();
  double elapsed = 0;
  double elapsed1 = 0;
  double elapsed2 = 0;
  double elapsed3 = 0;
  double elapsed4 = 0;

  spmv_csr_serial( num_rows, ptr, indices, data, x, y);
  
  //warmingup for dense_discrete
  elapsed = spmv_cuda_dense_discrete( num_rows, x, nnz, matrix, y_warmingup);
  //1) pass the full matrix via discrete memory, and regular MV
  for (i=0; i<num_runs; i++) elapsed1 += spmv_cuda_dense_discrete( num_rows, x, nnz, matrix, y_dense);
	
  //warmingup for csr_discrete
  elapsed = spmv_cuda_csr_discrete( num_rows, x, nnz, matrix, y_warmingup);
  //2) pass the csr format of the matrix via discrete memery, and sparseMV	
  for (i=0; i<num_runs; i++) elapsed2 += spmv_cuda_csr_discrete( num_rows, x, nnz, matrix, y_csr);
  
  //warmingup for unified memory
  elapsed = spmv_cuda_unified( num_rows, x, nnz, matrix, y_warmingup);
  //3) full matrix, pass indexes of non-zero elements, unified memory
  for (i=0; i<num_runs; i++) elapsed3 += spmv_cuda_unified( num_rows, x, nnz, matrix, y_unified);
  
  //warmingup for unified_count
  elapsed = spmv_cuda_unified_count( num_rows, x, nnz, matrix, y_warmingup);
  //4) full matrix on unified memory, pass the index of non-zero elements, but not the element themselves.  	
  for (i=0; i<num_runs; i++) elapsed4 += spmv_cuda_unified_count( num_rows, x, nnz, matrix, y_unified_count);
  //elapsed = (read_timer_ms() - elapsed)/num_runs;
  
  printf("%d,%f,%f,%f,%f\n", nnz, elapsed1/num_runs, elapsed2/num_runs, elapsed3/num_runs, elapsed4/num_runs);
  //printf("Spmv (csr) (%d): time: %0.2fms\n", nnz, elapsed2/num_runs);
  //printf("Spmv (unified) (%d): time: %0.2fms\n", nnz, elapsed3/num_runs);
  //printf("Spmv (unified_count) (%d): time: %0.2fms\n", nnz, elapsed4/num_runs);
  //printf("check_dense:%f\n",check(y,y_csr,num_rows));
  //printf("check_csr:%f\n",check(y,y_dense,num_rows));
  //printf("check_unified:%f\n",check(y,y_unified,num_rows));
  //printf("check_unified_count:%f\n",check(y,y_unified_count,num_rows));
  free(ptr);
  free(indices);
  free(data);
  free(x);
  free(y);
  free(y_csr);
  free(y_dense);
  return 0;
}
