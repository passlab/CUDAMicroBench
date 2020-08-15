// Experimental test input for Accelerator directives
//  simplest scalar*vector operations
// Liao 1/15/2013
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "SpMM_csc.h"
#include <time.h>

double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

int num_rows = 100;
int nnzA = 1024;
int nnzB = 1024;

void zero(REAL *A, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = 0.0;
    }
}

void init_data(REAL *data, int *indices, REAL *matrix)
{
  int tmp = 0;
	for (int i = 0; i < num_rows; i++) {
	  for (int j = 0; j < num_rows; j++) {
      if(matrix[i*num_rows+j] != 0) {
        data[tmp] = matrix[i*num_rows+j];
        indices[tmp] = j;
        tmp++;}
    }
  }
}

void init_data_csc(REAL *data, int *indices, REAL *matrix)
{
  int tmp = 0;
	for (int j = 0; j < num_rows; j++) {
	  for (int i = 0; i < num_rows; i++) {
      if(matrix[i*num_rows+j] != 0) {
        data[tmp] = matrix[i*num_rows+j];
        indices[tmp] = i;
        tmp++;}
    }
  }
}


void init_matrix(REAL * matrix, int nnz) {
    REAL d[num_rows*num_rows];
    int i,j,n,a,b,t;
    REAL A[num_rows][num_rows];
    srand(time(NULL));
    n=num_rows*num_rows;
    for (i=0;i<n;i++) d[i]=i;
    for (i=n;i>0;i--) {
        a=i-1;b=rand()%i;
        if (a!=b) {t=d[a];d[a]=d[b];d[b]=t;}
    }
 
    for (i=0;i<num_rows;i++) {
        for (j=0;j<num_rows;j++) {
            A[i][j]=(d[i*num_rows+j]>=nnz)?0:(REAL)(drand48()+1);
            matrix[i*num_rows+j] = A[i][j];
        }
    }
}

void init_ptr(int *ptr, REAL * matrix, int nnz)
{
  int tmp = 0;
  ptr[num_rows] = nnz;
  ptr[0] = 0;
  float * non_zero_elements;
  non_zero_elements = (float *) malloc(num_rows * sizeof(float));
	for (int i = 0; i < num_rows; i++) {
    int tmp = 0;
	  for (int j = 0; j < num_rows; j++) {
      if(matrix[i*num_rows+j] != 0) {
        tmp++;
      }
    non_zero_elements[i] = tmp;
    }
  }
  for (int i = 1; i<num_rows; i++){
    ptr[i] = ptr[i-1]+ non_zero_elements[i-1];
  }
}

void init_ptr_csc(int *ptr, REAL * matrix, int nnz)
{

  float * matrixT;
  matrixT = (float *) malloc(num_rows* num_rows * sizeof(float));
  for (int m = 0; m < num_rows; m++) {
		for (int n = 0; n < num_rows; n++) {
			matrixT[m*num_rows+n]=matrix[n*num_rows+m];
		}
	}
  int tmp = 0;
  ptr[num_rows] = nnz;
  ptr[0] = 0;
  float * non_zero_elements;
  non_zero_elements = (float *) malloc(num_rows * sizeof(float));
	for (int i = 0; i < num_rows; i++) {
    int tmp = 0;
	  for (int j = 0; j < num_rows; j++) {
      if(matrixT[i*num_rows+j] != 0) {
        tmp++;
      }
    non_zero_elements[i] = tmp;
    }
  }
  for (int i = 1; i<num_rows; i++){
    ptr[i] = ptr[i-1]+ non_zero_elements[i-1];
  }
}

void spmv_csc_serial( const int num_rows, const int *ptrA, const int * indicesA, const float *dataA, const int *ptrB, const int * indicesB, const float *dataB, float *result_serial, int nnzA, int nnzB){
    for(int row = 0; row < num_rows; row++){
        int row_start = ptrA[row];
        int row_end = ptrA[row+1];
        for(int column = 0; column < num_rows; column++){
            int column_start = ptrB[column];
            int column_end = ptrB[column+1];
            float dot = 0;
            for ( int i = row_start; i < row_end; i++) {
                for(int j = column_start; j < column_end; j++) {
                    if(indicesA[i] == indicesB[j]){
                          dot += dataA[i] * dataB[j];
                    }
                }
            }
         result_serial[row*num_rows+column] = dot;
        }
    }
}

void matmul_serial(float *A, float *B, float *C) {
	  float dummy = 0;
    int i,j,k;
    volatile float temp;
    for (i = 0; i < num_rows; i++) {
        for (j = 0; j < num_rows; j++) {
            temp = 0;
            for (k = 0; k < num_rows; k++) {
                temp += (A[i * num_rows + k] * B[k * num_rows + j]);
            }
            C[i * num_rows + j] = temp;
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

void printArray(int *pflMat, int M)
{
	for(int idxM = 0; idxM < M; idxM++)
	{
			printf("%d\t",pflMat[idxM]);
	}
	printf("\n");
}

void printArray1(REAL *pflMat, int M)
{
	for(int idxM = 0; idxM < M; idxM++)
	{
			printf("%f\t",pflMat[idxM]);
	}
	printf("\n");
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
    return diffsum;
}

int main(int argc, char *argv[])
{
  int *ptrA, * indicesA;
  int *ptrB, * indicesB;
  float * dataA, * dataB, * result_serial, *result_cuda, *result_serial_normal;
  float *matrixA, *matrixB;
  
  ptrA = (int *) malloc((num_rows+1) * sizeof(int));
  indicesA = (int *) malloc(nnzA * sizeof(int));
  dataA = (float *) malloc(nnzA * sizeof(float));
  
  ptrB = (int *) malloc((num_rows+1) * sizeof(int));
  indicesB = (int *) malloc(nnzB * sizeof(int));
  dataB = (float *) malloc(nnzB * sizeof(float));
  

  result_serial = (float *) malloc(num_rows * num_rows * sizeof(float));
  result_cuda = (float *) malloc(num_rows * num_rows * sizeof(float));
  result_serial_normal = (float *) malloc(num_rows * num_rows * sizeof(float));

  matrixA = (float *) malloc(num_rows * num_rows * sizeof(float));
  matrixB = (float *) malloc(num_rows * num_rows * sizeof(float));
  
  srand48(1<<12);
  init_matrix(matrixA,nnzA);
  init_matrix(matrixB,nnzB);
   
  init_data(dataA,indicesA, matrixA);
  init_data_csc(dataB,indicesB, matrixB);
  init_ptr(ptrA, matrixA,nnzA);
  init_ptr_csc(ptrB, matrixB,nnzB);

  int i;
  int num_runs = 5;
  /* cuda version */
  double elapsed = read_timer_ms();
  //for (i=0; i<num_runs; i++) 
  spmv_csc_serial( num_rows, ptrA, indicesA, dataA, ptrB, indicesB, dataB, result_cuda, nnzA,nnzB);
  matmul_serial( matrixA, matrixB, result_serial_normal);
  //for (i=0; i<num_runs; i++) 
  spmm_csr_cuda( num_rows, ptrA, indicesA, dataA, ptrB, indicesB, dataB, result_serial, nnzA,nnzB);
  elapsed = (read_timer_ms() - elapsed)/num_runs;

  printf("check(serial versions):%f\n",check(result_serial,result_serial_normal,(num_rows*num_rows)));
  printf("check(serial vs cuda):%f\n",check(result_cuda,result_serial_normal,(num_rows*num_rows)));

  free(ptrA);
  free(indicesA);
  free(dataA);
  free(ptrB);
  free(indicesB);
  free(dataB);

  free(result_serial);
  free(result_cuda);
  free(result_serial_normal);
  free(matrixA);
  free(matrixB);
  return 0;
}
