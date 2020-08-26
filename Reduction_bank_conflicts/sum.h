#define REAL float
#define ThreadsPerBlock 256
#define VEC_LEN 1024000 //use a fixed number for now

#ifdef __cplusplus
extern "C" {
#endif
extern void axpy_cuda(int n, REAL *x, REAL *result);
#ifdef __cplusplus
}
#endif
