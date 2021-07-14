//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#define REAL float


#ifdef __cplusplus
extern "C" {
#endif
extern void matadd(float * h_flMat1, float * h_flMat2, int iMatSizeM, int iMatSizeN, float * h_flMatSum);
#ifdef __cplusplus
}
#endif
