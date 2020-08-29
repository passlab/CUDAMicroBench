#define REAL double

#ifdef __cplusplus
extern "C" {
#endif
extern void axpy_cuda(REAL *x, REAL * y, int n, REAL a);
extern void axpy_cuda_optimized(REAL* x, REAL* y, int n, REAL a);
#ifdef __cplusplus
}
#endif
