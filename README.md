# HPC: Matrix Multiplication Optimization

This repository is the result of doing the High-Performance Computing (HPC) course at the University. It aims to optimize the code and use HPC techniques in order to achieve the highes performance computation on the cluster.

The repository is structured in the following way:

* `01_Matrix_Multiplication_Optimization` - bare optimisation of the matrix multiplication. Includes cache awareness, 1D matrix transforming, optimised loop ordering and loop vectorization.
* `02_OpenMP` - parallelise the code from `01_Matrix_Multiplication_Optimization` using the `OpenMP` directives. `#pragma parallel for` and `#pragma taskloop` comparison.
* `03_CUDA_MPI` - distribute the matrix multiplication operation across multiple devices and use accelerator programming to achieve higher performance.

Each directory includes the benchmarks and a short report over the optimisation results.
