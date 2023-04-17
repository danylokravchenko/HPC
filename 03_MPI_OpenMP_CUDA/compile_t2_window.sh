ml purge
ml CUDA/7.5.18
nvcc -c T2_window.cu -o cuda_t2_window.o
ml intel/2019a
mpiicc -c T2_window.c -fopenmp -o mpi_t2_window.o
mpiicc mpi_t2_window.o cuda_t2_window.o -fopenmp -lcudart -o T2_window.o
