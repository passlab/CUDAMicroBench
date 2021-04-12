NVCC=nvcc

CUDAFLAGS= -arch=sm_86 --cudart=shared -rdc=true -Xcompiler -fopenmp -lpng

OPT= -g -G

RM=/bin/rm -f

all: Dynamic_Parallelism Non_Dynamic_Parallelism


main: Dynamic_Parallelism.o Non_Dynamic_Parallelism.o

		${NVCC} ${OPT} -o main Dynamic_Parallelism.o 
		${NVCC} ${OPT} -o main Non_Dynamic_Parallelism.o
        

Dynamic_Parallelism.o: Dynamic_Parallelism.cu Non_Dynamic_Parallelism.cu

		$(NVCC) ${OPT} $(CUDAFLAGS) -c Dynamic_Parallelism.cu
		$(NVCC) ${OPT} $(CUDAFLAGS) -c Non_Dynamic_Parallelism.cu

Dynamic_Parallelism: Dynamic_Parallelism.o Non_Dynamic_Parallelism.o

		${NVCC} ${CUDAFLAGS} -o Dynamic_Parallelism Dynamic_Parallelism.o 
		${NVCC} ${CUDAFLAGS} -o Non_Dynamic_Parallelism Non_Dynamic_Parallelism.o 

clean:

		${RM} *.o Dynamic_Parallelism
		${RM} *.o Non_Dynamic_Parallelism