#define REAL double

#ifdef __cplusplus
extern "C" {
#endif
extern void axpy_cuda(REAL *x, REAL * y, int n, REAL a);
//extern void axpy_cuda_unified(REAL* x, REAL* y, int n, REAL a);
extern void axpy_cuda_optimized(REAL* x, REAL* y, int n, REAL a);
extern __global__ void axpy_cudakernel_part1(REAL* x, REAL* y, int n, REAL a);
extern void axpy_cpu_part2(REAL* x, REAL* y, int n, REAL a) ;
extern __global__ void axpy_cudakernel_part3(REAL* x, REAL* y, int n, REAL a);
extern void axpy_cpu_part4(REAL* x, REAL* y, int n, REAL a);
#ifdef __cplusplus
}
#endif