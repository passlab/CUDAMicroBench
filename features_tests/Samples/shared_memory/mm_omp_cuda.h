#define REAL double

#ifdef __cplusplus
extern "C" {
#endif
extern void mm_kernel(REAL*, REAL*, REAL*, int);
extern void mm_kernel_shmem(REAL*, REAL*, REAL*, int);
#ifdef __cplusplus
}
#endif
