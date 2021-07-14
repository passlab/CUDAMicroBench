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
#include <png.h>
#include <string.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>





using namespace std;

#define threhhold -1
#define DIFF_DWELL -1
#define MAX_DWELL 2048
#define BSX 64
#define BSY 16
#define MAX_DEPTH 16
#define neutral (MAX_DWELL + 1)


//nv git
struct complex {
    __host__ __device__ complex(float re, float im = 0) {
        this->re = re;
        this->im = im;
    }
    /** real and imaginary part */
    float re, im;
};

// operator overloads for complex numbers
inline __host__ __device__ complex operator+
(const complex &a, const complex &b) {
	return complex(a.re + b.re, a.im + b.im);
}
inline __host__ __device__ complex operator-
(const complex &a) { return complex(-a.re, -a.im); }
inline __host__ __device__ complex operator-
(const complex &a, const complex &b) {
	return complex(a.re - b.re, a.im - b.im);
}
inline __host__ __device__ complex operator*
(const complex &a, const complex &b) {
	return complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);
}
inline __host__ __device__ float abs2(const complex &a) {
	return a.re * a.re + a.im * a.im;
}
inline __host__ __device__ complex operator/
(const complex &a, const complex &b) {
	float invabs2 = 1 / abs2(b);
	return complex((a.re * b.re + a.im * b.im) * invabs2,
								 (a.im * b.re - b.im * a.re) * invabs2);
}  // operator/

/**
* from nv dev post
mariani_silver(rectangle)
  if (border(rectangle) has common dwell)
        fill rectangle with common dwell
    else if (rectangle size < threshold)
      per-pixel evaluation of the rectangle
    else
      for each sub_rectangle in subdivide(rectangle)
          mariani_silver(sub_rectangle)/
*/
struct pixel {
    complex z;
    int color;
};
/** a useful function to compute the number of threads 
copied from 24-25*/
int divup(int x, int y){
    return x / y + (x % y ? 1 : 0);
}


__device__ int border_check(complex* grid, int w, int h, int* dwells){
    int x_block = blockDim.x * blockIdx.x;
    int y_block = blockDim.y * blockIdx.y;
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    /**  nm 
    if (x_block == 0 || x_block == w - 1) {
        //think complex
        dwells[y * x] = render();
    }
    */
}

__device__ int get_dwell_eq(int d1, int d2) {
    if(d1 == d2){
    return 1;
    }
    else if (d1 == neutral || d2 == neutral) {
        return min(d1, d2);
    }
    return -1;
}
__device__ int render(int x, int y, int w, int h, complex cmin, complex cmax) {
    complex dc = cmax - cmin;
    float fx = (float)x / w;
    float fy = (float)y / h;
    complex c = cmin + complex (fx * dc.re, fy * dc.im);
    complex z = c;
    int dwell = 0;
    while (dwell < MAX_DWELL && abs2(z) <= 4) {
        z = z * z + c;
        dwell++;
    }
    return dwell;
}
__global__ void compute(int* dwells, int w, int h, complex cmin, complex cmax) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < w && y < h) {
        dwells[y * w + x] = render(x, y, w, h, cmin, cmax);
    }

}
/** gets the color, given the dwell (on host) */

#define CUT_DWELL (MAX_DWELL / 4)
void dwell_color(int* r, int* g, int* b, int dwell) {
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
            *r = *g = 0;
            *b = 128 + dwell * 127 / (CUT_DWELL);
        }
        else {
            // from blue to white for the second half
            *r = 255;
            *b = *g = (dwell - CUT_DWELL) * 255 / (MAX_DWELL - CUT_DWELL);
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

#define H (8 * 2000)
#define W (8 * 2000)
#define IMAGE_PATH "./mandelbrot.png"
int main(int argc, char **argv) {
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
    dim3 blocks(BSX, BSY), grid(divup(w, blocks.x), divup(h, blocks.y));
    double start;
    double end;
    start = omp_get_wtime();
    compute<<< grid, blocks>>>(dwellsD, w, h, cmin, cmax);
    cudaThreadSynchronize();
    end = omp_get_wtime();
    printf("Work took %f seconds\n", end - start);
    cudaMemcpy(dwellsH, dwellsD, dwells_size, cudaMemcpyDeviceToHost);

    save_image(IMAGE_PATH, dwellsH, w, h);

    cudaFree(dwellsD);
    free(dwellsH);
    return 0;
}