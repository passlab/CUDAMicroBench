// Experimental test input for Accelerator directives
//  simplest scalar*vector operations
// Liao 1/15/2013
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

int num_rows = 1000;
int nnz = 400;

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

/* initialize a vector with random floating point numbers */
void init_vector(REAL *vector, int m)
{
	for (int i = 0; i<m; i++) {
    vector[i] = (REAL)drand48();//(float)rand()/(float)(RAND_MAX/10.0);
  }
}



void init_matrix(REAL * matrix) {
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

void init_ptr(int *ptr, REAL * matrix)
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
  float * data, * x, *y, *y_cuda, *y_normal;
  float *matrix;
  
  ptr = (int *) malloc((num_rows+1) * sizeof(int));
  indices = (int *) malloc(nnz * sizeof(int));
  data = (float *) malloc(nnz * sizeof(float));
  x = (float *) malloc(num_rows * sizeof(float));
  y = (float *) malloc(num_rows * sizeof(float));
  y_cuda = (float *) malloc(num_rows * sizeof(float));
  y_normal = (float *) malloc(num_rows * sizeof(float));
  matrix = (float *) malloc(num_rows * num_rows * sizeof(float));
  
  srand48(1<<12);
  init_matrix(matrix);
  init_vector(x, num_rows);
  init_data(data,indices, matrix);
  init_ptr(ptr, matrix);

  int i;
  int num_runs = 5;
  /* cuda version */
  double elapsed = read_timer_ms();
  for (i=0; i<num_runs; i++) spmv_csr_serial( num_rows, ptr, indices, data, x, y);
  for (i=0; i<num_runs; i++) spmv_cuda( num_rows, ptr, indices, data, x, y_cuda, nnz, matrix, y_normal);
  elapsed = (read_timer_ms() - elapsed)/num_runs;

  printf("check:%f\n",check(y,y_cuda,num_rows));
  printf("check1:%f\n",check(y,y_normal,num_rows));

  free(ptr);
  free(indices);
  free(data);
  free(x);
  free(y);
  free(y_cuda);
  free(y_normal);
  return 0;
}
