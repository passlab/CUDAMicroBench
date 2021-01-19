	#include <stdio.h>
	#include <math.h>
	#include <cmath>
	#include <stdlib.h>
	#include <omp.h>
	#include <vector>

	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"


	#include <stdio.h>

	void sort(int *values, int n) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n - i -1; j++) {
				if (values[j] > values[j + 1]) {
					int temp = values[j];
					values[j] = values[j + 1];
					values[j + 1] = temp;
				}
			}
		}
		return;

	}


	__device__ int isPrimeF(int n) {
		int i;
		bool isPrime = true;

		// 0 and 1 are not prime numbers
		if (n == 0 || n == 1) {
			isPrime = false;
		}
		else {
			for (i = 2; i <= n / 2; ++i) {
				if (n % i == 0) {
					isPrime = false;
					break;
				}
			}
		}
		return isPrime;
	}


	__global__ void saxpy(float *y, float *x, int n, float a) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i < n) {
			y[i] = a * x[i] + y[i];
		}
		int j = 0;
		for (int q = 0; q < 1000000000; q++) {
			bool temp;
			while (j < 5313434234) {
				temp = isPrimeF((i - 4123453125) * (i - 4123453125));
				j++;
			}
			if(temp){
				double re = i;
			
			}
		}
	}

	#define N 300000000 
	#define K 10000
int main() {
	std::vector<double> memcpyt;
	std::vector<double> ttl;
	std::vector<double> kernelt;
	std::vector<double> hostt;

	for (int q = 0; q < 1; q++) {
		cudaStream_t stream1;
		cudaError_t result;
		result = cudaStreamCreate(&stream1);


		int size = N * sizeof(float);
		int blockS = (N + 255) / 256;
		//generating random array
		int values[K];
		for (int i = 0; i < K; i++) {
			values[i] = (int)rand() % (K * 4);
		}
		float* x = (float*)(malloc(size));
		float* y = (float*)(malloc(size));
		for (int i = 0; i < N; i++) {
			y[i] = rand() % (N * 4);
			x[i] = rand() % (N * 4);
		}


		float* d_x, * d_y;
		cudaMalloc(&d_x, size);
		cudaMalloc(&d_y, size);

		double start, memStart, kernelStart, hostStart, end, memEnd, kernelEnd, hostEnd;
		start = omp_get_wtime();

		//Async mem copy to Device
		cudaMemcpyAsync(d_x, x, size, cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(d_y, y, size, cudaMemcpyHostToDevice, stream1);


		//timers
		memEnd = omp_get_wtime();
		kernelStart = omp_get_wtime();

		//kernel launch
		saxpy << <blockS, 256, 1, stream1 >> > (y, x, N, 2.0);
		//cudaThreadSynchronize();
		//timers
		kernelEnd = omp_get_wtime();
		hostStart = omp_get_wtime();

		//host launch
		sort(values, K);
		

		//timers
		end = omp_get_wtime();
		ttl.push_back(end - start);
		memcpyt.push_back(memEnd - start);
		kernelt.push_back(kernelEnd - kernelStart);
		hostt.push_back(end - hostStart);
		cudaFree(d_y);
		free(x);
		free(y);
		cudaStreamDestroy(stream1);
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

		//display times
		printf("MemCpy time is %f \n", memTime);
		printf("Kernel execution time is %f \n", time - (hostTime + memTime));
		printf("Host execution time is %f \n", hostTime);
		printf("Total execution time is %f \n", time);


		return 0;

}