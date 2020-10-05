#define REAL float

#ifdef __cplusplus
extern "C" {
#endif
extern void spmv_cuda_csr_discrete(const int num_rows, const REAL * x, int nnz, REAL* matrix, REAL *y, double * elapsed);
extern void spmv_cuda_dense_discrete(const int num_rows, const REAL * x, int nnz, REAL* matrix, REAL *y, double * elapsed);
extern void init_csr(int *ptr, REAL *data, int *indices, REAL *matrix, int num_rows, int nnz);
extern void init_index(int * row, int * column, REAL *matrix, int num_rows);
extern void spmv_cuda_unified(const int num_rows, const REAL * x, int nnz, REAL* matrix, REAL *y, double * elapsed);
extern void spmv_cuda_unified_count(const int num_rows, const REAL * x, int nnz, REAL* matrix, REAL *y, double * elapsed);
extern void init_index_count(int * row_nnz, int * row, int * column, REAL *matrix, int num_rows);
//extern void init_ptr(int *ptr, REAL * matrix, int num_rows, int nnz);
extern double read_timer_ms();
#ifdef __cplusplus
}
#endif
