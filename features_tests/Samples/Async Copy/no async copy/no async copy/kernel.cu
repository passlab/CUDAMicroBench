#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void sort(int* values, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n - i - 1; j++) {
			if (values[j] > values[j + 1]) {
				int temp = values[j];
				values[j] = values[j + 1];
				values[j + 1] = temp;
			}
		}
	}
	return;

}


__global__ void saxpy(float* y, float* x, int n, float a) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		y[i] = a * x[i] + y[i];
	}
}



#define N 100000000
#define K 10000
int main() {
	std::vector<double> memcpyt;
	std::vector<double> ttl;
	std::vector<double> kernelt;
	std::vector<double> hostt;
	
	for (int q = 0; q < 30; q++) {
		int size = N * sizeof(float);
		int blockS = (N + 255) / 256;
		//generating random array
		int values[K];
		for (int i = 0; i < K; i++) {
			values[i] = (int)rand() % (N * 4);
		}
		float* x = (float*)(malloc(size));
		float* y = (float*)(malloc(size));
		for (int i = 0; i < N; i++) {
			y[i] = rand() % (K * 4);
			x[i] = rand() % (K * 4);
		}


		float* d_x, * d_y;
		cudaMalloc(&d_x, size);
		cudaMalloc(&d_y, size);

		double start, memStart, kernelStart, hostStart, end, memEnd, kernelEnd, hostEnd;
		start = omp_get_wtime();
		//standard mem cpy
		cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);


		//timers
		memEnd = omp_get_wtime();
		kernelStart = omp_get_wtime();
		//kernel launch
		saxpy << <blockS, 256 >> > (y, x, N, 2.0);
		//cudaThreadSynchronize();
		//timers
		kernelEnd = omp_get_wtime();
		hostStart = omp_get_wtime();

		//host launch
		sort(values, K);

		//timers
		hostEnd = omp_get_wtime();
		end = omp_get_wtime();
		ttl.push_back(end - start);
		memcpyt.push_back(memEnd - start);
		kernelt.push_back(kernelEnd - kernelStart);
		hostt.push_back(hostEnd - hostStart);
		cudaFree(d_x);
		cudaFree(d_y);
		free(x);
		free(y);
		printf("iteration \n");
	}
	double sum[4] = { 0.0, 0.0, 0.0, 0.0 };
	for (int i = 0; i < hostt.size(); i++) {
		sum[0] += hostt[i];
		sum[1] += memcpyt[i];
		sum[2] += kernelt[i];
		sum[3] += ttl[i];

	}
	for (int i = 0; i < 4; i++) {
		sum[i] = (sum[i] / (double)hostt.size());
	}

	double memTime = sum[1];
	double kernelTime = sum[2];
	double hostTime = sum[0];
	double time = sum[3];
	printf("MemCpy time is %f \n", memTime);
	printf("Kernel execution time is %f \n", kernelTime);
	printf("Host execution time is %f \n", hostTime);
	printf("Total execution time is %f \n", time);



	return 0;

}