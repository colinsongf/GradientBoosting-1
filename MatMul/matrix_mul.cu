#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <iostream>

#define BLOCK_SIZE 32

void matMulCpu(float* a, float* b, float* c, int s1, int s2, int s3)
{
	for (int i = 0; i < s1; i++)
	{
		for (int j = 0; j < s3; j++)
		{
			float ans = 0;
			for (int k = 0; k < s2; k++)
			{
				ans += a[i * s2 + k] * b[k * s3 + j];
			}
			c[i * s3 + j] = ans;
		}
	}
}

__global__ void matMulGpu(float* a, float* b, float* c, int s1, int s2, int s3)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < s1 && j < s3)
	{
		float ans = 0;
		for (int k = 0; k < s2; k++)
		{
			ans += a[i * s2 + k] * b[k * s3 + j];
		}
		c[i * s3 + j] = ans;
	}
}

void fillMat(float* a, int s)
{
	for (int i = 0; i < s; i++)
	{
		a[i] = rand() % 100;
	}
}

float calcSumCpu(float* a, int s)
{
	float ans = 0;
	for (int i = 0; i < s; i++)
	{
		ans += a[i];
	}
	return ans;
}

int main()
{
	//freopen("out.txt", "w", stdout);
	int iterations = 2;
	int size = 1000;
	srand(time(NULL));
	float* a = (float*)malloc(size * size * sizeof(float));
	float* b = (float*)malloc(size * size * sizeof(float));
	float* c = (float*)malloc(size * size * sizeof(float));

	float* a_device;
	float* b_device;
	float* c_device;
	cudaMalloc(&a_device, size * size * sizeof(float));
	cudaMalloc(&b_device, size * size * sizeof(float));
	cudaMalloc(&c_device, size * size * sizeof(float));

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(2 + size * size / (1 + BLOCK_SIZE), 2 + size * size / (1 + BLOCK_SIZE));

	float sum_h;
	float sum_d;

	clock_t time_h = 0;
	clock_t time_d = 0;


	for (int i = 0; i < iterations; i++)
	{
		fillMat(a, size * size);
		fillMat(b, size * size);
		cudaMemcpy(a_device, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(b_device, b, size * size * sizeof(float), cudaMemcpyHostToDevice);

		clock_t t1 = clock();
		matMulCpu(a, b, c, size, size, size);
		t1 = clock() - t1;
		time_h += t1;

		t1 = clock();
		matMulGpu<<<grid, block>>>(a_device, b_device, c_device, size, size, size);
		cudaDeviceSynchronize();
		t1 = clock() - t1;
		time_d += t1;

		sum_h = calcSumCpu(c, size * size);
		cudaMemcpy(c, c_device, size * size * sizeof(float), cudaMemcpyDeviceToHost);
		sum_d = calcSumCpu(c, size * size);
		printf("host: %f device: %f ", sum_h, sum_d);
		if (sum_h == sum_d)
		{
			printf("OK\n");
		}
	}

	printf("time_h: %f time_d: %f\n\n", (float)time_h / CLOCKS_PER_SEC, (float)time_d / CLOCKS_PER_SEC);

	//fclose(stdout);
	free(a);
	free(b);
	free(c);
	cudaFree(a_device);
	cudaFree(b_device);
	cudaFree(c_device);

    return 0;
}
