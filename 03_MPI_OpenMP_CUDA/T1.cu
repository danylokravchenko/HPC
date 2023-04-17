#define GPU_THREADS_PER_BLOCK 1024
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>

__global__ void matrix_mult_kernel(int *a, int *b, int *c, int width);

void checkCudaError(const char *message);

void init(int *input, int length);

void print_matrix(int *matrix, int size);

int main(int argc, char * argv[])
{
	unsigned int N = 0, runs;
	double time;
	if (argc == 2 && isdigit(argv[1][0])) {
        N = atoi(argv[1]);
    }else {
        printf("USAGE\n   %s [SIZE] \n", argv[0]);
        return 0;
    }
	int *h_a, *d_a;
	int *h_b, *d_b;
	int *h_c, *d_c;
	int data_length = N * N * sizeof(int);

	// allocate the matrix h_a, h_b, h_c at the host memory
	h_a = (int *)malloc(data_length);
	h_b = (int *)malloc(data_length);
	h_c = (int *)malloc(data_length);

	init(h_a, N * N);
	init(h_b, N * N);

	// allocate the matrix d_a, d_b at the device memory
	cudaMalloc(&d_a, data_length);
	cudaMalloc(&d_b, data_length);

	// check for errors during allocations
	checkCudaError("Failed to allocate matrices on the device memory");

	// larger block size allow for better memory usage and higher arithmetic intensity
	// 32 = sqrt(N); We have 1024 threads.
	int threadsDim = sqrt(GPU_THREADS_PER_BLOCK);
	// direct initialization
	dim3 threadsPerBlock(threadsDim, threadsDim, 1);
	// add threadsPerBlock.x - 1 to each nominator to ensure that
	// the trancation always leads to the correct integer
    dim3 numBlocks(
		(N + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(N + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

	runs = time = 0;

	while (runs < 5) {
		// allocate the matrix d_c at the device memory
		cudaMalloc(&d_c, data_length);
		// check for errors during allocations
		checkCudaError("Failed to allocate matrices on the device memory");

		// create CUDA event timers to measure the time
		// default timers will not work because it will stall the device
		cudaEvent_t t1, t2;
		cudaEventCreate(&t1);
		cudaEventCreate(&t2);

		// record the event of moving the data and starting the kernel
		cudaEventRecord(t1);

		// copy input matrix h_a and h_b to the device memory
		cudaMemcpy(d_a, h_a, data_length, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, data_length, cudaMemcpyHostToDevice);
		// check for errors during copying matrices to the device memory
		checkCudaError("Failed to copy matrices to the device memory");

		// launch matrix multiplication kernel
		matrix_mult_kernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
		// check for errors during launching the matrix multiplication kernel
		checkCudaError("Failed to launch the matrix multiplication kernel");

		// wait for results
		cudaDeviceSynchronize();

		// copy output matrix h_c to the host memory
		cudaMemcpy(h_c, d_c, data_length, cudaMemcpyDeviceToHost);
		// check for errors during copying the output to the host memory
		checkCudaError("Failed to copy the output to the host memory");

		// record the event of finishing the kernel and copying the data back
		cudaEventRecord(t2);

		// measure the time taken t2 - t1
		cudaEventSynchronize(t2);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, t1, t2);
		// printf("Elapsed time: %f ms\n", elapsedTime);
		time+= elapsedTime;
		runs++;

	}

	printf("MATRIX SIZE: %i, RUNS: %i\n", N, runs);

	printf("Mean execution time: %f ms\n", (time/runs));

	// free all alocated memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	// check for errors during freeing the alocated memory
	checkCudaError("Failed to free the alocated memory");

	free(h_a);
	free(h_b);
	free(h_c);
}

void checkCudaError(const char *message) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		printf("%s: %s\n", message, cudaGetErrorString(err));
		exit(-1);
	}
}

void init(int *input, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		input[i] = rand() % 5;
	}
}

void print_matrix(int *matrix, int size)
{
	printf("Matrix items: \n");
	int i, j;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
			printf("%d,", matrix[i * size + j]);
		printf("\n");
	}
}

__global__ void matrix_mult_kernel(int *a, int *b, int *c, int width)
{
	// the naive matrix multiplication version
	// the method takes a portion of the data assigned to it by the GPU
	// and performs the multiplication on this data only

	// The kernel uses thread indexing to distribute
	// the computation across the available threads on the GPU.

	// That's why it's important to figure out
	// the correct indices of the assigned portion of the data
	// thread identification.
	// kernel gives us i and j indices of the data to operate on
	int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;

	// check that only threads responsable for the corresponding indices are used for computation
	// to prevent overwriting of the output
    if (i < width && j < width)
    {
		// the kernel performs the k loop of naive implementation
        for (int k = 0; k < width; k++)
        {
            sum += a[i * width + k] * b[k * width + j];
        }
        c[i * width + j] = sum;
    }
}
