
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

cudaError_t matMulWithCuda(double* c, const double* a, const double* b, unsigned int size);

__global__ void matMulKernel(double* c, const double* a, const double* b, int N)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	double sum = 0;
	for (int k = 0; k < N; k++) {
		sum += a[i * N + k] * b[k * N + j];
	}
	c[i * N + j] = sum;
}

int main()
{
	for (int N = 1; N <= 32; N = N * 2) {
		printf("SIZE = %d \n", N);
		double* c = (double*)malloc(sizeof(double) * N * N);
		double* a = (double*)malloc(sizeof(double) * N * N);
		double* b = (double*)malloc(sizeof(double) * N * N);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				a[i * N + j] = (float)i + j;
				b[i * N + j] = (float)i - j;
			}
		}
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaError_t cudaStatus = matMulWithCuda(c, a, b, N);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		/*for (int i = 0; i < N * N; i++) {
			if (i % N == 0) printf("\n");
			printf("%lf ", c[i]);
		}*/

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		free(a);
		free(b);
		free(c);
	}
	return 0;
}

cudaError_t matMulWithCuda(double* c, const double* a, const double* b, unsigned int ldim)
{
	double* dev_a = 0;
	double* dev_b = 0;
	double* dev_c = 0;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int matrixDim = ldim * ldim;
	cudaEventRecord(start);
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
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\tCOPY TIME: %lfms\n", milliseconds);
	// Launch a kernel on the GPU with one thread for each element.
	cudaEventRecord(start);
	for (int count = 0; count < 10; count++) {
		matMulKernel << <1, threadsPerBlock >> > (dev_c, dev_a, dev_b, ldim);
	}
	cudaEventRecord(stop);

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
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\tCORE TIME: %lfms\n", milliseconds / 10);
Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}


