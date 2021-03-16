default: mm_omp_cuda

clean: 
	rm -rf ${OBJS} *.log *.out

mm_omp_cuda: mm_omp_cuda.c mm_omp_cuda.h mm_kernel.cu
	nvcc mm_omp_cuda.c mm_kernel.cu -o mm_omp_cuda.out

