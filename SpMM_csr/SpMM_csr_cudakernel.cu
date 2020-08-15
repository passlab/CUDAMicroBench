#include "SpMM_csr.h"

__global__ void spmm_csr_kernel(const int num_rows, const int *ptrA, const int * indicesA, const REAL *dataA, const int *ptrB, const int * indicesB, const REAL *dataB,  REAL* result, int nnzA, int nnzB)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < num_rows){

        int row_start = ptrA[row];
        int row_end = ptrA[row+1];

        for(int k =0; k<num_rows; k++){ //iterate over B column
          float dot = 0;
          for ( int i = row_start; i < row_end; i++) {
            //int colNum = k;  //The col of the element
            for(int j = 0; j < nnzB; j++) { //nnz should be number of non-zero element of B
              if (indicesB[j] == k && j >= ptrB[indicesA[i]] && j < ptrB[indicesA[i]+1]) {
                dot += dataA[i] * dataB[j];
              }
            }
          }
        result[row*num_rows+k] = dot;
        }
    }
}

__global__ void matvec_cudakernel_1perThread(REAL* matrix, REAL* vector, REAL *y, int num_rows)
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

__global__ void matvec_cudakernel_1perThread_check_and_compute(REAL* matrix, REAL* vector, REAL *y, int num_rows)
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


void spmv_cuda(const int num_rows, const int *ptrA, const int * indicesA, const REAL *dataA, const int *ptrB, const int * indicesB, const REAL *dataB,  REAL* result, int nnzA, int nnzB) {
  int *d_ptrA, * d_indicesA, *d_ptrB, * d_indicesB;
  REAL * d_dataA, * d_dataB, * d_result;

  cudaMalloc(&d_ptrA, (num_rows+1)*sizeof(int));
  cudaMalloc(&d_indicesA, nnzA*sizeof(int));

  cudaMalloc(&d_ptrB, (num_rows+1)*sizeof(int));
  cudaMalloc(&d_indicesB, nnzB*sizeof(int));


  cudaMalloc(&d_dataA, nnzA*sizeof(REAL));
  cudaMalloc(&d_dataB, nnzB*sizeof(REAL));

  cudaMalloc(&d_result, num_rows * num_rows * sizeof(REAL));

  cudaMemcpy(d_ptrA, ptrA, (num_rows+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indicesA, indicesA, nnzA*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dataA, dataA, nnzA*sizeof(REAL), cudaMemcpyHostToDevice);

  cudaMemcpy(d_ptrB, ptrB, (num_rows+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indicesB, indicesB, nnzB*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dataB, dataB, nnzB*sizeof(REAL), cudaMemcpyHostToDevice);

  spmm_csr_kernel<<<256,256>>>(num_rows,d_ptrA, d_indicesA, d_dataA, d_ptrB, d_indicesB, d_dataB, d_result, nnzA, nnzB);
  cudaMemcpy(result, d_result, num_rows*num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);
  //cudaMemcpy(y_normal, d_y_normal, num_rows*sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaFree(d_ptrA);
  cudaFree(d_indicesA);
  cudaFree(d_dataA);
  cudaFree(d_ptrB);
  cudaFree(d_indicesB);
  cudaFree(d_dataB);
  cudaFree(d_result);

}
