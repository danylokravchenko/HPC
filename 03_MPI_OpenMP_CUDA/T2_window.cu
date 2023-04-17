#include <stdio.h>

__global__ void matrix_multiplication (double * d_a, double * d_b, double *d_c, int width)
{
	int k; double sum = 0;
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	//printf("d_c %x\n",d_c);
	if(col < width && row < width)
	{
		for (k = 0; k < width; k++)
			sum += d_a[row * width + k] * d_b[k * width + col];
		d_c[row * width + col] = sum;
		//printf("%lf\n",d_c[row * width + col]);
	//	printf("d_c %x\n",d_c);
	}

}

extern "C" int invoke_cuda_matrix_multiplication(double * h_a, double *h_b, double *h_c, int size, double ** d_a, double ** d_b, double ** d_c)
{

	// Allocate the matrix d_a, d_b, and d_c at the device memory
    int data_length = size * size * sizeof(double);
    cudaMalloc(d_a, data_length);
    cudaMalloc(d_b, data_length);
    cudaMalloc(d_c, data_length);

 	cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
    printf("cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
    }

    // Copy input matrix h_a and h_b to the device memory
    cudaMemcpy(*d_a, h_a, data_length, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b, h_b, data_length, cudaMemcpyHostToDevice);

    err = cudaGetLastError();
 	if ( cudaSuccess != err )
    {
    printf("cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
    }

    // Launch your matrix multiplication kernel - hint do not wait for results and do not free cuda resources see the function below (get_results)
    dim3 block_size(32, 32);
    dim3 grid_size((size + block_size.x - 1) / block_size.x, (size + block_size.y - 1) / block_size.y);
    matrix_multiplication<<<grid_size, block_size>>>(*d_a, *d_b, *d_c, size);

 	err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
    printf("cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
    }

	return 0;
}

extern "C" void get_results(double *h_c, double ** d_a, double ** d_b, double ** d_c,int data_length)
{

	cudaError err;
	//printf("data length %d\n",data_length);
	cudaDeviceSynchronize();
	//printf("get d_c %x\n",*d_c);
	cudaMemcpy(h_c, *d_c,data_length, cudaMemcpyDeviceToHost);
	//printf("%lf %lf %lf\n", h_c[0],h_c[1],h_c[2]);
  err = cudaGetLastError();
  if ( cudaSuccess != err )
  {
          printf("cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
  }
  cudaFree(*d_c);
  cudaFree(*d_a);
  cudaFree(*d_b);
}
