//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#define REAL float

#ifdef __cplusplus
extern "C" {
#endif
extern void warpDivergenceTest_cuda(REAL* x, REAL* y, REAL *warp_divergence, REAL *no_warp_divergence, int n);
#ifdef __cplusplus
}
#endif
