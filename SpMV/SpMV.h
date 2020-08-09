#define REAL float

#ifdef __cplusplus
extern "C" {
#endif
extern void spmv_cuda(const int num_rows, const int *ptr, const int * indices, const float *data, const float * x, float *y, int nnz, REAL* matrix, float *y_normal);
#ifdef __cplusplus
}
#endif
