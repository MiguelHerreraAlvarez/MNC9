
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <stdio.h>
#include <cstdio>
#include <cstdlib>

#define tam_blocks 2
cudaError_t addWithCuda(double* c, const double* a, const double* b, unsigned int size);

__global__ void addKernel(double* c, const double* a, const double* b, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0;
    for (int k = 0; k < N; k++)
        sum += a[i * N + k] * b[k * N + j];
    c[i * N + j] = sum;
}

int main()
{
    for (int k = 64; k <= 1024; k = k * 2) {
        const int arraySize = k;
        double* A = (double*)malloc(sizeof(double) * arraySize * arraySize);
        double* B = (double*)malloc(sizeof(double) * arraySize * arraySize);
        double* C = (double*)malloc(sizeof(double) * arraySize * arraySize);

        for (int i = 0; i < arraySize; i++)
        {
            for (int j = 0; j < arraySize; j++) {
                A[arraySize * i + j] = i + j;
                B[arraySize * i + j] = i - j;
            }
        }
        cudaError_t cudaStatus;
        // Add vectors in parallel.
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) {
            cudaStatus = addWithCuda(C, A, B, arraySize);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }
        printf("Tiempo CUDA: %f milisegundos, N = %d\n", ((float)milliseconds / 10.) / CLOCKS_PER_SEC, arraySize);

        /*if (arraySize == 64) {
            printf("Diagonal de la matriz de tamaño 64*64");
            for (int i = 0; i < arraySize; i++)
            {
                printf(" %lf ", C[arraySize * i + i]);
            }
        }*/
        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }

    }
    
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(double* c, const double* a, const double* b, unsigned int size)
{
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    int total_size = size * size;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, total_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, total_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, total_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, total_size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, total_size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    if (size > 32) {
        dim3 threadsPerBlock(size / 32, size / 32);
        dim3 numBlocks(tam_blocks, tam_blocks);
        // Launch a kernel on the GPU with one thread for each element.
        addKernel << <numBlocks, threadsPerBlock >> > (dev_c, dev_a, dev_b, size);
    }
    else {
        dim3 threadsPerBlock(size, size);
        dim3 numBlocks(tam_blocks, tam_blocks);
        // Launch a kernel on the GPU with one thread for each element.
        addKernel << <numBlocks, threadsPerBlock >> > (dev_c, dev_a, dev_b, size);
    }
    
    
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
    cudaStatus = cudaMemcpy(c, dev_c, total_size * sizeof(double), cudaMemcpyDeviceToHost);
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

