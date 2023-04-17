#define GPU_THREADS_PER_BLOCK 1024
#include <math.h>
#include <stdio.h>

void checkCudaError(const char *message);

struct CUDA_Events
{
    // create CUDA event timers to measure the time
    cudaEvent_t t1;
    cudaEvent_t t2;
    float elapsedTime;
};

typedef struct CUDA_Events CUDA_Events;

CUDA_Events *cudaEvents = (CUDA_Events*)malloc(sizeof(CUDA_Events));

__global__ void matrix_mult_kernel(double *d_a, double *d_b, double *d_c, int width)
{
    // kernel gives us i and j indices of the data to operate on
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k;
    double sum = 0;

    // check that only threads responsable for the corresponding indices are used for computation
    // to prevent overwriting of the output
    if (i < width && j < width)
    {
        // the kernel performs the k loop of naive implementation
        for (k = 0; k < width; k++)
            sum += d_a[i * width + k] * d_b[k * width + j];
        d_c[i * width + j] = sum;
    }
}

extern "C" int invoke_cuda_matrix_multiplication(double *h_a, double *h_b,
    double *h_c, int size, double **d_a, double **d_b, double **d_c)
{
    int data_length = size * size * sizeof(double);
    // check time t1
	cudaEventCreate(&cudaEvents->t1);
	// record the event of moving the data and starting the kernel
	cudaEventRecord(cudaEvents->t1);
    // allocate the matrix d_a, d_b, and d_c at the device memory
	cudaMalloc(d_a, data_length);
	cudaMalloc(d_b, data_length);
    cudaMalloc(d_c, data_length);
    // check for errors during allocations
	checkCudaError("Failed to allocate matrices on the device memory");

    // copy input matrix h_a and h_b to the device memory
	cudaMemcpy(*d_a, h_a, data_length, cudaMemcpyHostToDevice);
	cudaMemcpy(*d_b, h_b, data_length, cudaMemcpyHostToDevice);
	// check for errors during copying matrices to the device memory
	checkCudaError("Failed to copy matrices to the device memory");

    // larger block size allow for better memory usage and higher arithmetic intensity
	// 32 = sqrt(N); We have 1024 threads.
	int threadsDim = sqrt(GPU_THREADS_PER_BLOCK);
	// direct initialization
	dim3 threadsPerBlock(threadsDim, threadsDim, 1);
	// add threadsPerBlock.x - 1 to each nominator to ensure that
	// the trancation always leads to the correct integer
    dim3 numBlocks(
        (size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (size + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

    // launch matrix multiplication kernel
	matrix_mult_kernel<<<numBlocks, threadsPerBlock>>>(*d_a, *d_b, *d_c, size);
	// check for errors during launching the matrix multiplication kernel
	checkCudaError("Failed to launch the matrix multiplication kernel");

    return 0;
}

extern "C" float get_kernel_results(
    double *h_c, double **d_a, double **d_b, double **d_c, int size)
{
    int data_length = size * size * sizeof(double);
    cudaDeviceSynchronize();
    // copy output matrix h_c to the host memory
    cudaMemcpy(h_c, *d_c, data_length, cudaMemcpyDeviceToHost);
	// check for errors during copying the output to the host memory
	checkCudaError("Failed to copy the output to the host memory");

    cudaFree(*d_c);
    cudaFree(*d_a);
    cudaFree(*d_b);
    // check for errors during freeing the alocated memory
	checkCudaError("Failed to free the alocated memory");

    // check t2 event
    cudaEventCreate(&cudaEvents->t2);
	// record the event of finishing the kernel and copying the data back
	cudaEventRecord(cudaEvents->t2);
	// measure the execution time
	cudaEventSynchronize(cudaEvents->t2);
    float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, cudaEvents->t1, cudaEvents->t2);

    free(cudaEvents);

    return elapsedTime;
}


void checkCudaError(const char *message) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		printf("%s: %s - %s\n", message, cudaGetErrorName(err), cudaGetErrorString(err));
		exit(-1);
	}
}