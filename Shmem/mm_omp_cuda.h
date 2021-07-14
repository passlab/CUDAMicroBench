//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#define REAL double

#ifdef __cplusplus
extern "C" {
#endif
extern void mm_kernel(REAL*, REAL*, REAL*, int);
extern void mm_kernel_shmem(REAL*, REAL*, REAL*, int);
#ifdef __cplusplus
}
#endif
