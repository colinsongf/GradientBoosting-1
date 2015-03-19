#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <iostream>

#define BLOCK_SIZE 32

void matMulCpu(double* a, double* b, double* c, int s1, int s2, int s3)
{
	for (int i = 0; i < s1; i++)
	{
		for (int j = 0; j < s3; j++)
		{
			double ans = 0;
			for (int k = 0; k < s2; k++)
			{
				ans += a[i * s2 + k] * b[k * s3 + j];
			}
			c[i * s3 + j] = ans;
		}
	}
}

__global__ void matMulGpu(double* a, double* b, double* c, int s1, int s2, int s3)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < s1 && j < s3)
	{
		double ans = 0;
		for (int k = 0; k < s2; k++)
		{
			ans += a[i * s2 + k] * b[k * s3 + j];
		}
		c[i * s3 + j] = ans;
	}
}

void fillMat(double* a, int s)
{
	for (int i = 0; i < s; i++)
	{
		a[i] = rand() % 100;
	}
}

double calcSumCpu(double* a, int s)
{
	double ans = 0;
	for (int i = 0; i < s; i++)
	{
		ans += a[i];
	}
	return ans;
}

int main()
{
	freopen("out.txt", "w", stdout);
	printf("dd");
	int iterations = 10;
	int size = 100;
	srand(time(NULL));
	double* a = (double*)malloc(size * size * sizeof(double));
	double* b = (double*)malloc(size * size * sizeof(double));
	double* c = (double*)malloc(size * size * sizeof(double));

	double* a_device;
	double* b_device;
	double* c_device;
	cudaMalloc(&a_device, size * size * sizeof(double));
	cudaMalloc(&b_device, size * size * sizeof(double));
	cudaMalloc(&c_device, size * size * sizeof(double));

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(2 + size * size / (1 + BLOCK_SIZE), 2 + size * size / (1 + BLOCK_SIZE));

	double sum_h;
	double sum_d;

	for (int i = 0; i < iterations; i++)
	{
		fillMat(a, size * size);
		fillMat(b, size * size);
		cudaMemcpy(a_device, &a, size * size * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(b_device, &b, size * size * sizeof(double), cudaMemcpyHostToDevice);
		matMulCpu(a, b, c, size, size, size);
		matMulGpu<<<grid, block>>>(a_device, b_device, c_device, size, size, size);
		cudaDeviceSynchronize();
		sum_h = calcSumCpu(c, size * size);
		cudaMemcpy(c, c_device, size * size * sizeof(double), cudaMemcpyDeviceToHost);
		sum_d = calcSumCpu(c, size * size);
		printf("host: %f device: %f\n", sum_h, sum_d);
	}

	fclose(stdout);
	free(a);
	free(b);
	free(c);
	cudaFree(a_device);
	cudaFree(b_device);
	cudaFree(c_device);

    return 0;
}
