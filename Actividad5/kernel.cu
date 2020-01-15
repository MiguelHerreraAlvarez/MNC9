
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

cudaError_t matrixMultiplicationWithCuda(double* c, const double* a, const double* b, unsigned int size);

__global__ void matrixMultiplicationKernel(double* c, const double* a, const double* b, int N)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	double sum = 0;
	for (int k = 0; k < N; k++) {
		sum += a[i * N + k] * b[k * N + j];
	}
	c[i * N + j] = sum;
}

__global__ void showThreadInfo(double* c, const double* a, const double* b)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int k = blockDim.x;
	int m = k * j + i;
	printf("Indice de hilo: %d, Indice de bloque: %d, dim de bloque: %d, Calculo: %lf * %lf = %lf\n", i, j, k, a[m], b[m], a[m] * b[m]);
}

int main()
{
	int N = 3;
	//double *a = (double*)malloc(sizeof(double) * N*N);
	//double *b = (double*)malloc(sizeof(double) * N*N);
	double* c = (double*)malloc(sizeof(double) * N * N);
	double a[9] = { 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0 };
	double b[9] = { 0.0, -1.0, -2.0, 1.0, 0.0, -1.0, 2.0, 1.0, 0.0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = matrixMultiplicationWithCuda(c, a, b, N);
	//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	for (int i = 0; i < N * N; i++) {
		if (i % N == 0) printf("\n");
		printf("%lf ", c[i]);
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	//free(a);
	//free(b);
	free(c);
	return 0;
}

cudaError_t matrixMultiplicationWithCuda(double* c, const double* a, const double* b, unsigned int ldim)
{
	double* dev_a = 0;
	double* dev_b = 0;
	double* dev_c = 0;
	cudaError_t cudaStatus;

	int matrixDim = ldim * ldim;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, matrixDim * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, matrixDim * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, matrixDim * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, matrixDim * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, matrixDim * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 threadsPerBlock(ldim, ldim);

	// Launch a kernel on the GPU with one thread for each element.
	matrixMultiplicationKernel << <1, threadsPerBlock >> > (dev_c, dev_a, dev_b, ldim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, matrixDim * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
