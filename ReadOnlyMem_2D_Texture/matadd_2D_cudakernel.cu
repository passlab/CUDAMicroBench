//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#include "matadd_2D.h"
#include"cuda_runtime.h"  
#include"device_launch_parameters.h"  
#include <stdio.h>

#define BLOCK_SIZE 16

texture<float,2>texMatrixA;
texture<float,2>texMatrixB;

//constant memory
__constant__ int cons_M;
__constant__ int cons_N;

__global__ void add_warmingup(float * d_matrixA, float * d_matrixB, float *d_Result, int d_M, int d_N)  
{  
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if(tidx<d_M && tidy<d_N) {
        d_Result[tidx * d_N + tidy] = d_matrixA[tidx * d_N + tidy] + d_matrixB[tidx * d_N + tidy];
    }
}  

__global__ void add(float * d_matrixA, float * d_matrixB, float *d_Result, int d_M, int d_N)  
{  
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if(tidx<d_M && tidy<d_N) {
        d_Result[tidx * d_N + tidy] = d_matrixA[tidx * d_N + tidy] + d_matrixB[tidx * d_N + tidy];
    }
}  

__global__ void add_const(float * d_matrixA, float * d_matrixB, float *d_Result)  
{  
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if(tidx<cons_M && tidy<cons_N) {
        d_Result[tidx * cons_N + tidy] = d_matrixA[tidx * cons_N + tidy] + d_matrixB[tidx * cons_N + tidy];
    }
}  


__global__ static void add_texture(float *d_Result, int d_M, int d_N)
{
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if(tidx<d_M && tidy<d_N) {
        float u = tex2D(texMatrixA,tidx,tidy);
        float v = tex2D(texMatrixB,tidx,tidy);
        d_Result[tidx * d_N + tidy] = u + v;
    }
}

__global__ static void add_texture_constant(float *d_Result)
{
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if(tidx<cons_M && tidy<cons_N) {
        float u = tex2D(texMatrixA,tidx,tidy);
        float v = tex2D(texMatrixB,tidx,tidy);
        d_Result[tidx * cons_N+ tidy] = u + v;
    }
}


void matadd(float * h_matrixA, float * h_matrixB, int M, int N, float * h_result) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    float *d_matrixA = NULL, *d_matrixB = NULL, *d_result = NULL;
    cudaMalloc(&d_matrixA, M * N * sizeof(float));
    cudaMalloc(&d_matrixB, M * N * sizeof(float));
    cudaMalloc(&d_result, M * N * sizeof(float));

    cudaMemcpy(d_matrixA, h_matrixA, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, h_matrixB, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTexture2D(0, texMatrixA, d_matrixA, channelDesc, N, M, M * sizeof(float));
    cudaBindTexture2D(0, texMatrixB, d_matrixB, channelDesc, N, M, M * sizeof(float));

    cudaMemcpyToSymbol(cons_M,&M,sizeof(float),0);
    cudaMemcpyToSymbol(cons_N,&N,sizeof(float),0);

    dim3 blocks(1,1,1);
    dim3 threadsperblock(BLOCK_SIZE,BLOCK_SIZE,1);
    blocks.x=((M/BLOCK_SIZE) + (((M)%BLOCK_SIZE)==0?0:1));
    blocks.y=((N/BLOCK_SIZE) + (((N)%BLOCK_SIZE)==0?0:1));

    add_warmingup<<<blocks,threadsperblock>>>(d_matrixA,d_matrixB,d_result,M,N);
    cudaDeviceSynchronize();
    add<<<blocks,threadsperblock>>>(d_matrixA,d_matrixB,d_result,M,N);
    cudaDeviceSynchronize();
    add_const<<<blocks,threadsperblock>>>(d_matrixA,d_matrixB,d_result);
    cudaDeviceSynchronize();
    add_texture<<<blocks,threadsperblock>>>(d_result,M,N);
    cudaDeviceSynchronize();
    add_texture_constant<<<blocks,threadsperblock>>>(d_result);
    cudaDeviceSynchronize();

    cudaDeviceSynchronize();
    cudaMemcpy(h_result,d_result,M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaUnbindTexture(texMatrixA);
    cudaUnbindTexture(texMatrixB);

    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_result);
}
