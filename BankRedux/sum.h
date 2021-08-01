//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#define REAL float
#define ThreadsPerBlock 256
#define VEC_LEN 1024000 //use a fixed number for now

#ifdef __cplusplus
extern "C" {
#endif
extern void sum_cuda(int n, REAL *x, REAL *result);
#ifdef __cplusplus
}
#endif
