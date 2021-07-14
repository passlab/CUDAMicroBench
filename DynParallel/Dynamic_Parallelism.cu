//******************************************************************************************************************//
// Copyright (c) 2021, University of North Carolina at Charlotte
// and Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: (BSD-3-Clause)
//*****************************************************************************************************************//
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <png.h>
#include <string.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <zlib.h>









using namespace std;

#define threhhold -1
#define DIFF_DWELL -1
#define MAX_DWELL 2048
#define BSX 64
#define BSY 16
#define MAX_DEPTH 16
#define neutral (MAX_DWELL + 1)
/** region below which do per-pixel */
#define MIN_SIZE 64
/** subdivision factor along each axis */
#define SUBDIV 16
/** subdivision when launched from host */
#define INIT_SUBDIV 64

#define NEUT_DWELL (MAX_DWELL + 1)


/** gets the color, given the dwell (on host) */

#define CUT_DWELL (MAX_DWELL / MAX_DEPTH)
void dwell_color(int* r, int* g, int* b, float dwell) {
    // black for the Mandelbrot set
    if (dwell >= MAX_DWELL) {
        *r = *g = *b = 0;
    }
    else {
        // cut at zero
        if (dwell < 0)
            dwell = 0;
        if (dwell <= CUT_DWELL) {
            // from black to blue the first half
            *r = 120;
            *g = 0;
            *b = 128 + dwell * 127 / (CUT_DWELL);
        }
        else {
            // from blue to white for the second half
            *b = 20;
            *r = *g = (dwell - CUT_DWELL) * 255 / (MAX_DWELL - CUT_DWELL);
        }
    }
}  // dwell_color

/** from nv page*/
void save_image(const char* filename, int* dwells, int w, int h) {
    png_bytep row;

    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    // exception handling
    setjmp(png_jmpbuf(png_ptr));
    png_init_io(png_ptr, fp);
    // write header (8 bit colour depth)
    png_set_IHDR(png_ptr, info_ptr, w, h,
        8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    // set title
    png_text title_text;
    title_text.compression = PNG_TEXT_COMPRESSION_NONE;
    title_text.key = "Title";
    title_text.text = "Mandelbrot set, per-pixel";
    png_set_text(png_ptr, info_ptr, &title_text, 1);
    png_write_info(png_ptr, info_ptr);

    // write image data
    row = (png_bytep)malloc(3 * w * sizeof(png_byte));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int r, g, b;
            dwell_color(&r, &g, &b, dwells[y * w + x]);
            row[3 * x + 0] = (png_byte)r;
            row[3 * x + 1] = (png_byte)g;
            row[3 * x + 2] = (png_byte)b;
        }
        png_write_row(png_ptr, row);
    }
    png_write_end(png_ptr, NULL);

    fclose(fp);
    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    free(row);
}  // save_image



//nv git
struct complex {
    __host__ __device__ complex(float re, float im = 0) {
        this->re = re;
        this->im = im;
    }
    /** real and imaginary part */
    float re, im;
};

__host__ __device__ complex absolute(complex x) {
    x.re = abs(x.re);
    x.im = abs(x.im);
    return x;
}
__host__ __device__ complex logC(complex x) {
    x.re = log(x.re);
    x.im = log(x.im);
    return x;
}

// operator overloads for complex numbers
inline __host__ __device__ complex operator+
(const complex& a, const complex& b) {
    return complex(a.re + b.re, a.im + b.im);
}
inline __host__ __device__ complex operator-
(const complex& a) {
    return complex(-a.re, -a.im);
}
inline __host__ __device__ complex operator-
(const complex& a, const complex& b) {
    return complex(a.re - b.re, a.im - b.im);
}
inline __host__ __device__ complex operator*
(const complex& a, const complex& b) {
    return complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);
}
inline __host__ __device__ float abs2(const complex& a) {
    return a.re * a.re + a.im * a.im;
}
inline __host__ __device__ complex operator/
(const complex& a, const complex& b) {
    float invabs2 = 1 / abs2(b);
    return complex((a.re * b.re + a.im * b.im) * invabs2,
        (a.im * b.re - b.im * a.re) * invabs2);
} 

// operator/
/** a useful function to compute the number of threads
copied from 24-25*/
__device__ __host__ int divup(int x, int y) {
    return x / y + (x % y ? 1 : 0);
}

__device__ int get_dwell_eq(int d1, int d2) {
    if (d1 == d2) {
        return d1;
    }
    else if (d1 == NEUT_DWELL || d2 == NEUT_DWELL) {
        return min(d1, d2);
    }
    return -1;
}

__device__ int dwell_function(int x, int y, int w, int h, complex cmin, complex cmax) {
    complex dc = cmax - cmin;
    float fx = (float)x / w;
    float fy = (float)y / h;
    complex c = cmin + complex(fx * dc.re, fy * dc.im);
    int dwell = 0;
    complex z = c;
    while (dwell < MAX_DWELL && abs2(z) < 2 * 2) {
        z = z * z + c;
        dwell++;
    }
    return dwell;
}

__global__ void pixel_calc(int* dwells, int w, int h, int x0, int y0, complex cmin, complex cmax, int d) {
    int x = threadIdx.x + blockDim.x * blockIdx.x; 
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < d && y < d) {
        x += x0, y += y0;
        dwells[y * w + x] = dwell_function(x, y, w, h, cmin, cmax);
    }
}

__global__ void dwell_fill_k(int* dwells, int w, int x0, int y0, int d, int dwell) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < d && y < d) {
        x += x0, y += y0;
        dwells[y * w + x] = dwell;
    }
}
//nv blog
__device__ int border_dwell(int w, int h, complex cmin, complex cmax, int x0, int y0, int d) {
    // check whether all boundary pixels have the same dwell
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int bs = blockDim.x * blockDim.y;
    int comm_dwell = NEUT_DWELL;
    // for all boundary pixels, distributed across threads
    for (int r = tid; r < d; r += bs) {
        // for each boundary: b = 0 is east, then counter-clockwise
        for (int b = 0; b < 4; b++) {
            int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
            int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
            int dwell = dwell_function(x, y, w, h, cmin, cmax);
            comm_dwell = get_dwell_eq(comm_dwell, dwell);
        }
    }  // for all boundary pixels
    // reduce across threads in the block
    __shared__ int ldwells[BSX * BSY];
    int nt = min(d, BSX * BSY);
    if (tid < nt)
        ldwells[tid] = comm_dwell;
    __syncthreads();
    for (; nt > 1; nt /= 2) {
        if (tid < nt / 2)
            ldwells[tid] = get_dwell_eq(ldwells[tid], ldwells[tid + nt / 2]);
        __syncthreads();
    }
    //printf("%i", ldwells[0]);
    return ldwells[0];
}


__global__ void mandelbrot_block_k(int* dwells, int w, int h, complex cmin, complex cmax, int x0, int y0,
    int d, int depth) {
    x0 += d * blockIdx.x, y0 += d * blockIdx.y;
    int comm_dwell = border_dwell(w, h, cmin, cmax, x0, y0, d);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (comm_dwell != DIFF_DWELL) {
            // uniform dwell, just fill
            dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
            dwell_fill_k << <grid, bs >> > (dwells, w, x0, y0, d, comm_dwell);
        }
        else if (depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
            // subdivide recursively
            dim3 bs(blockDim.x, blockDim.y), grid(SUBDIV, SUBDIV);
            mandelbrot_block_k << <grid, bs >> >
                (dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth + 1);
        }
        else {
            // leaf, per-pixel kernel
            dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
            //maybe broke since not treating as kernel launch
            pixel_calc<<<grid, bs>>>(dwells, w, h, x0, y0, cmin, cmax, d);
            //__syncthreads();
        }
    }
}  // mandelbrot_block_k



#define H (8 * 2000)
#define W (8 * 2000)
#define IMAGE_PATH "./mandelbrot_dp.png"
int main(int argc, char** argv) {
    int w = W;
    int h = H;
    size_t dwells_size = w * h * sizeof(int);
    int* dwellsD;
    int* dwellsH;
    cudaMalloc((void**)&dwellsD, dwells_size);
    dwellsH = (int*)malloc(dwells_size);
    complex cmin = complex(-1.5, -1);
    complex cmax = complex(.5, 1);
    //kernel dims
    dim3 blocks(BSX, BSY), grid(INIT_SUBDIV,  INIT_SUBDIV);
    double start;
    double end;
    start = omp_get_wtime();
    mandelbrot_block_k << < grid, blocks >> > (dwellsD, w, h, cmin, cmax, 0, 0, W / INIT_SUBDIV, 1);
    cudaThreadSynchronize();
    end = omp_get_wtime();
    printf("Work took %f seconds\n", end - start);
    cudaMemcpy(dwellsH, dwellsD, dwells_size, cudaMemcpyDeviceToHost);
    printf("saving image....");
    cudaFree(dwellsD);
    save_image(IMAGE_PATH, dwellsH, w, h);

    free(dwellsH);
    return 0;
}
