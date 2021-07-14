//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#include "SpMV.h"
#include <stdio.h>
#include <stdlib.h>

//2 csr-format of the matrix, copy csr formatted-matrix via discrete memory
__global__ void spmv_csr(const int num_rows, const int *ptr, const int * indices, const REAL *data, const REAL * x, REAL *y)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < num_rows){
        REAL dot = 0;
        int row_start = ptr[row];
        int row_end = ptr[row+1];
       
        for(int n = row_start; n < row_end; n++){
           dot += data[n] * x[indices[n]];
        }
        y[row] = dot;
    }
}

// 1) full matrix, discrete memory to copy the full matrix
__global__ void spmv_dense(REAL* matrix, REAL* vector, REAL *y, int num_rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows) {
        y[i] = 0;
        REAL temp = 0.0;
        for (int j = 0; j < num_rows; j++)
            temp += matrix[i * num_rows + j] * vector[j];
        y[i] = temp;
    }
}

__global__ void spmv_dense_check_and_compute(REAL* matrix, REAL* vector, REAL *y, int num_rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows) {
        y[i] = 0;
        REAL temp = 0.0;
        for (int j = 0; j < num_rows; j++) {
            if (matrix[i * num_rows + j] != 0.0) 
                temp += matrix[i * num_rows + j] * vector[j];
        }
        y[i] = temp;
    }
}

// 3) kernels for full matrix stored in unified memory, data copied to GPU explicitly via cudaMemcpy are indexes of non-zero elements
__global__ void spmv_unified(REAL* matrix_unified, const int num_rows, const int *rowNum, const int * columnNum, const REAL * x, REAL *y, int nnz)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    REAL dot = 0;
    for (int n = 0; n < nnz; n++){
        if(rowNum[n] == row){
           dot += matrix_unified[row * num_rows + columnNum[n]]* x[columnNum[n]];
        }
    }
    y[row] = dot;
}

// 4) kernels for full matrix stored in unified memory, data copied to GPU explicitly via cudaMemcpy are indexes of non-zero elements, 
// and the column of the first non-zero element of each row
__global__ void spmv_unified_count(int * count, REAL* matrix_unified, const int num_rows, const int *rowNum, const int * columnNum, const REAL * x, REAL *y, int nnz)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int row_start = count[row];
    int row_end = count[row+1];

    REAL dot = 0;
    for (int n = row_start; n < row_end; n++){
      dot += matrix_unified[row * num_rows + columnNum[n]]* x[columnNum[n]];
    }
    y[row] = dot;
}

double spmv_cuda_dense_discrete(const int num_rows, const REAL * x, int nnz, REAL* matrix, REAL *y) {
  double elapsed1 = read_timer_ms();

  REAL * d_x, *d_matrix, *d_y;
  cudaMalloc(&d_x, num_rows*sizeof(REAL));
  cudaMalloc(&d_matrix, num_rows * num_rows * sizeof(REAL));
  cudaMalloc(&d_y, num_rows*sizeof(REAL));

  //timer start for H2D
  cudaMemcpy(d_x, x, num_rows*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix, matrix, num_rows*num_rows*sizeof(REAL), cudaMemcpyHostToDevice);
  //timer end for H2D
    
  //timer start for kernel
  spmv_dense_check_and_compute<<<256, 256>>>(d_matrix, d_x, d_y, num_rows);
  cudaDeviceSynchronize();
  //timer stop for kernel

 //timer start for D2H
  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);
    //timer stop for D2H

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_matrix);
  elapsed1 = (read_timer_ms() - elapsed1);
  return elapsed1;
}

double spmv_cuda_csr_discrete(const int num_rows, const REAL * x, int nnz, REAL* matrix, REAL *y) {

  int *ptr, * indices;
  float * data;
  
  ptr = (int *) malloc((num_rows+1) * sizeof(int));
  indices = (int *) malloc(nnz * sizeof(int));
  data = (float *) malloc(nnz * sizeof(float));
  
  init_csr(ptr, data,indices, matrix, num_rows,nnz);
  double elapsed1 = read_timer_ms();

  int *d_ptr, * d_indices;
  REAL * d_data, * d_x, *d_y;

  cudaMalloc(&d_ptr, (num_rows+1)*sizeof(int));
  cudaMalloc(&d_indices, nnz*sizeof(int));

  cudaMalloc(&d_data, nnz*sizeof(REAL));
  cudaMalloc(&d_x, num_rows*sizeof(REAL));

  cudaMalloc(&d_y, num_rows*sizeof(REAL));

  cudaMemcpy(d_ptr, ptr, (num_rows+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, indices, nnz*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, data, nnz*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, num_rows*sizeof(REAL), cudaMemcpyHostToDevice);


  spmv_csr<<<256,256>>>(num_rows,d_ptr, d_indices, d_data, d_x, d_y);
  cudaDeviceSynchronize();
  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);
  free(data);
  free(ptr);
  free(indices);

  cudaFree(d_ptr);
  cudaFree(d_indices);
  cudaFree(d_data);
  cudaFree(d_x);
  cudaFree(d_y);
  elapsed1 = (read_timer_ms() - elapsed1);
  return elapsed1;

}


double spmv_cuda_unified(const int num_rows, const REAL * x, int nnz, REAL* matrix, REAL *y) {
  int *row, * column;
  row = (int *) malloc(nnz * sizeof(int));
  column= (int *) malloc(nnz * sizeof(int));

  double elapsed1 = read_timer_ms();
  REAL *matrix_unified;
  cudaMallocManaged(&matrix_unified, num_rows*num_rows*sizeof(REAL));
  memcpy(matrix_unified, matrix, num_rows*num_rows*sizeof(REAL));
  REAL * d_x, *d_y;
    
  init_index(row , column, matrix, num_rows);
  //double elapsed1 = read_timer_ms();

  int *d_row, * d_column;

  cudaMalloc(&d_row, nnz*sizeof(int));
  cudaMalloc(&d_column, nnz*sizeof(int));
  cudaMalloc(&d_x, num_rows*sizeof(REAL));
  cudaMalloc(&d_y, num_rows*sizeof(REAL));

  cudaMemcpy(d_row, row, nnz*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_column, column, nnz*sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_x, x, num_rows*sizeof(REAL), cudaMemcpyHostToDevice);
  spmv_unified<<<256,256>>>(matrix_unified, num_rows,d_row, d_column, d_x, d_y, nnz);
  cudaDeviceSynchronize();
  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);

  free(row);
  free(column);
  cudaFree(d_row);
  cudaFree(d_column);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(matrix_unified);
  elapsed1 = (read_timer_ms() - elapsed1);
  return elapsed1;

}

double spmv_cuda_unified_count(const int num_rows, const REAL * x, int nnz, REAL* matrix, REAL *y) {
  int *row, * column, *count;
  row = (int *) malloc(nnz * sizeof(int));
  column= (int *) malloc(nnz * sizeof(int));
  count = (int *) malloc(num_rows * sizeof(int));


  double elapsed1 = read_timer_ms();
  REAL *matrix_unified;
  cudaMallocManaged(&matrix_unified, num_rows*num_rows*sizeof(REAL));
  memcpy(matrix_unified, matrix, num_rows*num_rows*sizeof(REAL));
  REAL * d_x, *d_y;
    
  init_index_count(count, row , column, matrix, num_rows);
  //double elapsed1 = read_timer_ms();

  int *d_row, * d_column, *d_count;

  cudaMalloc(&d_row, nnz*sizeof(int));
  cudaMalloc(&d_column, nnz*sizeof(int));
  cudaMalloc(&d_count, num_rows*sizeof(int));

  cudaMalloc(&d_x, num_rows*sizeof(REAL));
  cudaMalloc(&d_y, num_rows*sizeof(REAL));

  cudaMemcpy(d_row, row, nnz*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_column, column, nnz*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_count, count, num_rows*sizeof(int), cudaMemcpyHostToDevice);


  cudaMemcpy(d_x, x, num_rows*sizeof(REAL), cudaMemcpyHostToDevice);
  spmv_unified<<<256,256>>>(matrix_unified, num_rows,d_row, d_column, d_x, d_y, nnz);
  cudaDeviceSynchronize();
  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);

  free(row);
  free(column);
  free(count);
  cudaFree(d_row);
  cudaFree(d_column);
  cudaFree(count);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(matrix_unified);
  elapsed1 = (read_timer_ms() - elapsed1);
  return elapsed1;

}


