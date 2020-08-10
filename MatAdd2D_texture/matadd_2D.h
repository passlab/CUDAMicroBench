#define REAL float


#ifdef __cplusplus
extern "C" {
#endif
extern void matadd(float * h_flMat1, float * h_flMat2, int iMatSizeM, int iMatSizeN, float * h_flMatSum);
#ifdef __cplusplus
}
#endif
