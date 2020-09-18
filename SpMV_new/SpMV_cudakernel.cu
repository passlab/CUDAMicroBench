#include "SpMV.h"

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
        y[row] += dot;
    }
}

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

double warmingup_dense(const int num_rows, const REAL * x, int nnz, REAL* matrix, REAL *y) {
  double elapsed1 = read_timer_ms();

  REAL * d_x, *d_matrix, *d_y;
  cudaMalloc(&d_x, num_rows*sizeof(REAL));
  cudaMalloc(&d_matrix, num_rows * num_rows * sizeof(REAL));

  cudaMalloc(&d_y, num_rows*sizeof(REAL));

  cudaMemcpy(d_x, x, num_rows*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix, matrix, num_rows*num_rows*sizeof(REAL), cudaMemcpyHostToDevice);
  spmv_dense_check_and_compute<<<256, 256>>>(d_matrix, d_x, d_y, num_rows);
  cudaDeviceSynchronize();

  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_matrix);
  elapsed1 = (read_timer_ms() - elapsed1);
  return elapsed1;
}

double warmingup_csr(const int num_rows, const REAL * x, int nnz, REAL* matrix, REAL *y) {
  int *ptr, * indices;
  float * data;
  
  ptr = (int *) malloc((num_rows+1) * sizeof(int));
  indices = (int *) malloc(nnz * sizeof(int));
  data = (float *) malloc(nnz * sizeof(float));

  double elapsed1 = read_timer_ms();
  init_csr(ptr, data,indices, matrix, num_rows,nnz);
  //double elapsed1 = read_timer_ms();
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
  cudaFree(d_ptr);
  cudaFree(d_indices);
  cudaFree(d_data);
  cudaFree(d_x);
  cudaFree(d_y);
  elapsed1 = (read_timer_ms() - elapsed1);
  return elapsed1;
}



double spmv_cuda_dense_discrete(const int num_rows, const REAL * x, int nnz, REAL* matrix, REAL *y) {
  double elapsed1 = read_timer_ms();

  REAL * d_x, *d_matrix, *d_y;
  cudaMalloc(&d_x, num_rows*sizeof(REAL));
  cudaMalloc(&d_matrix, num_rows * num_rows * sizeof(REAL));

  cudaMalloc(&d_y, num_rows*sizeof(REAL));

  cudaMemcpy(d_x, x, num_rows*sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix, matrix, num_rows*num_rows*sizeof(REAL), cudaMemcpyHostToDevice);
  spmv_dense_check_and_compute<<<256, 256>>>(d_matrix, d_x, d_y, num_rows);
  cudaDeviceSynchronize();

  cudaMemcpy(y, d_y, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);

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
  
  double elapsed1 = read_timer_ms();
  init_csr(ptr, data,indices, matrix, num_rows,nnz);
  //double elapsed1 = read_timer_ms();

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
  cudaFree(d_ptr);
  cudaFree(d_indices);
  cudaFree(d_data);
  cudaFree(d_x);
  cudaFree(d_y);
  elapsed1 = (read_timer_ms() - elapsed1);
  return elapsed1;

}

