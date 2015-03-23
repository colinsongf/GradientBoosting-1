#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <iostream>
#include <cublas_v2.h>

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
	float* c_device_cublas;
	cudaMalloc(&a_device, size * size * sizeof(float));
	cudaMalloc(&b_device, size * size * sizeof(float));
	cudaMalloc(&c_device, size * size * sizeof(float));
	cudaMalloc(&c_device_cublas, size * size * sizeof(float));

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(1 + size / (1 + BLOCK_SIZE), 1 + size / (1 + BLOCK_SIZE));

	float sum_h;
	float sum_d;

	clock_t time_h = 0;
	clock_t time_d = 0;
	float time_d_event = 0;
	float time_d_cublas = 0;

	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle); 
	float alpha = 1.0f;
	float beta = 0.0f;

	// warmup
	fillMat(a, size * size);
	fillMat(b, size * size);
	cudaMemcpy(a_device, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_device, b, size * size * sizeof(float), cudaMemcpyHostToDevice);
	matMulGpu<<<grid, block>>>(a_device, b_device, c_device, size, size, size);
	cudaDeviceSynchronize();
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, b, size, a, size,
			&beta, c_device_cublas, size);
	cudaDeviceSynchronize();
	// done

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < iterations; i++)
	{
		fillMat(a, size * size);
		fillMat(b, size * size);
		cudaMemcpy(a_device, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(b_device, b, size * size * sizeof(float), cudaMemcpyHostToDevice);

		clock_t t1 = clock();
		float time;
		matMulCpu(a, b, c, size, size, size);
		t1 = clock() - t1;
		time_h += t1;

		t1 = clock();
		cudaEventRecord(start, 0);
		matMulGpu<<<grid, block>>>(a_device, b_device, c_device, size, size, size);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		t1 = clock() - t1;
		time_d += t1;

		cudaEventElapsedTime(&time, start, stop);
		time_d_event += (time / 1000.0);

		cudaEventRecord(start, 0);
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, b, size, a, size,
			&beta, c_device_cublas, size);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		time_d_cublas += (time / 1000.0);

		sum_h = calcSumCpu(c, size * size);
		cudaMemcpy(c, c_device, size * size * sizeof(float), cudaMemcpyDeviceToHost);
		sum_d = calcSumCpu(c, size * size);
		if (sum_h == sum_d)
		{
			printf("OK ");
		}
		printf("host: %f device: %f ", sum_h, sum_d);
		cudaMemcpy(c, c_device_cublas, size * size * sizeof(float), cudaMemcpyDeviceToHost);
		sum_d = calcSumCpu(c, size * size);
		if (sum_h == sum_d)
		{
			printf("OK ");
		}
		printf("cublas: %f\n", sum_d);
	}

	float time_h_secs = (float)time_h / CLOCKS_PER_SEC;
	float time_d_secs = (float)time_d / CLOCKS_PER_SEC;
	float profit = time_h_secs / time_d_secs;
	float profit_event = time_h_secs / time_d_event;
	float profit_cublas = time_h_secs / time_d_cublas;
	printf("profit: %f profit event: %f profit cublas %f time_h: %f time_d: %f\n\n", profit,
		profit_event, profit_cublas, time_h_secs, time_d_secs);

	//fclose(stdout);
	status = cublasDestroy(handle);
	free(a);
	free(b);
	free(c);
	cudaFree(a_device);
	cudaFree(b_device);
	cudaFree(c_device);

    return 0;
}
