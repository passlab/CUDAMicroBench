#define REAL float
#ifdef __cplusplus
extern "C" {
#endif
extern void LowAccessDensityTest_cuda(REAL* x, REAL* y, long int n, REAL a, int stride);
extern void LowAccessDensityTest_cuda_unified(REAL* x, REAL* y, long int n, REAL a, int stride);
#ifdef __cplusplus
}
#endif
