// Experimental test input for Accelerator directives
//  simplest scalar*vector operations
// Liao 1/15/2013
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

/* change this to do saxpy or daxpy : single precision or double precision*/
#define REAL float
#define VEC_LEN 102400000//use a fixed number for now
#define STRIDE 1024

/* zero out the entire vector */
void zero(REAL *A, long int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = 0.0;
    }
}

/* initialize a vector with random floating point numbers */
void init(REAL *A, long int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = (double)drand48();
    }
}

/*serial version */
void serial_kernel(REAL* x, REAL* y, long n, REAL a, int stride) {
  int i;
  for (i = 0; i < (n/stride); i++)
  {
    y[i] = a * x[i*stride];
  }
}

/*omp version */
void omp_kernel(REAL* x, REAL* y, long n, REAL a, int stride) {
  int i;
  #pragma omp parallel for shared(x,y,a,n,stride) private(i)  
  for (i = 0; i < (n/stride); i++)
  {
    y[i] = a * x[i*stride];
  }
}


/* compare two arrays and return percentage of difference */
REAL check(REAL*A, REAL*B, long int n)
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
  long int n;
  int stride = STRIDE;
  REAL *y_omp, *y, *x;
  REAL a = 123.456;

  n = VEC_LEN;
  fprintf(stderr, "Usage: %s <stride> [vec_len]\n", argv[0]);
  if (argc >= 2) {
    stride = atoi(argv[1]);
  }
  if (argc >= 3) {
    n = atoi(argv[2]);
  }

  // same input x
  x = (REAL *) malloc(n * sizeof(REAL));
  srand48(1<<12);
  init(x, n);

  // output for serial and omp version
  y  = (REAL *) malloc((n/stride) * sizeof(REAL));
  y_omp  = (REAL *) malloc((n/stride) * sizeof(REAL));


  // serial version as a reference
  serial_kernel(x, y, n, a, stride);

  int i;
  int num_runs = 100;

  /* OMP version */
  double elapsed = read_timer_ms();
  for (i=0; i<num_runs; i++) 
    omp_kernel(x, y_omp, n, a, stride);
  elapsed = (read_timer_ms() - elapsed)/num_runs;


  free(x);
  free(y);
  free(y_omp);
  return 0;
}
