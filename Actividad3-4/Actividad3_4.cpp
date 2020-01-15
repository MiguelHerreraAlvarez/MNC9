#include <cstdio>
#include <cstdlib>
#include <time.h>

#define n_iterations 100000

void MatrixMultiplication(double* A, double* B, double* C, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float sum = 0;
			for (int k = 0; k < N; k++)
				sum += A[i * N + k] * B[k * N + j];
			C[i * N + j] = sum;
		}
	}
}
void main() {
	clock_t start, stop;

	double* A;
	double* B;
	double* C;
	
	for (int i = 1; i <= 64 ; i*=2)
	{
		A = (double*)malloc(sizeof(double) * i * i);
		B = (double*)malloc(sizeof(double) * i * i);
		C = (double*)malloc(sizeof(double) * i * i);

		for (int k = 0; k < i; k++)
		{
			for (int j = 0; j < i; j++)
			{
				A[k * i + j] = k + j;
				B[k * i + j] = k - j;
			}
		}

		start = clock();

		for (int j = 0; j < n_iterations; j++)
		{
			MatrixMultiplication(A, B, C, i);
		}
		
		stop = clock();
		printf("SIZE = %d\tTIME: %f ms\n", i, ((float)(stop - start) / n_iterations) / CLOCKS_PER_SEC);

		if (i == 64) {
			printf("Diagonal");
			for (int k = 0; k < i; k++)
			{
				printf(" %.2f ", C[k * i + k]);
			}
		}
		
		free(A);
		free(B);
		free(C);
	}
}