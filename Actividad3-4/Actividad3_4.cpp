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
	int lengths[] = { 1,2,4,8,16,32,64 };
	for (int i = 0; i < sizeof(lengths) / sizeof(int); i++)
	{
		A = (double*)malloc(sizeof(double) * lengths[i] * lengths[i]);
		B = (double*)malloc(sizeof(double) * lengths[i] * lengths[i]);
		C = (double*)malloc(sizeof(double) * lengths[i] * lengths[i]);

		for (int k = 0; k < lengths[i]; k++)
		{
			for (int j = 0; j < lengths[i]; j++)
			{
				A[k * lengths[i] + j] = k + j;
				B[k * lengths[i] + j] = k - j;
			}
		}

		start = clock();

		for (int j = 0; j < n_iterations; j++)
		{
			MatrixMultiplication(A, B, C, lengths[i]);
		}
		
		stop = clock();
		printf("Tiempo secuencial: %f milisegundos para N=%d\n", ((float)(stop - start) / n_iterations) / CLOCKS_PER_SEC, lengths[i]);

		if (lengths[i] == 64) {
			printf("Diagonal de la matriz de tamaño 64*64");
			for (int k = 0; k < lengths[i]; k++)
			{
				printf(" %lf ", C[k * lengths[i] + k]);
			}
		}
		free(A);
		free(B);
		free(C);
	}
}