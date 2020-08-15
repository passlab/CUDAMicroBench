#define REAL float

#ifdef __cplusplus
extern "C" {
#endif
extern void spmm_csr_cuda(const int num_rows, const int *ptrA, const int * indicesA, const REAL *dataA, const int *ptrB, const int * indicesB, const REAL *dataB,  REAL* result, int nnzA, int nnzB);

extern void spmm_csc_cuda(const int num_rows, const int *ptrA, const int * indicesA, const REAL *dataA, const int *ptrB, const int * indicesB, const REAL *dataB,  REAL* result, int nnzA, int nnzB);

#ifdef __cplusplus
}
#endif
