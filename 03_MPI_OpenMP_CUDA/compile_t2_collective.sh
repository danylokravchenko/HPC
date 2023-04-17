ml purge
ml CUDA/7.5.18
nvcc -c T2_collective.cu -o cuda_t2_collective.o
ml intel/2019a
mpiicc -c T2_collective.c -fopenmp -o mpi_t2_collective.o
mpiicc mpi_t2_collective.o cuda_t2_collective.o -fopenmp -lcudart -o T2_collective.o
