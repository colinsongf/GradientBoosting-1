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

__global__ void matMulGpuShared(float* a, float* b, float* c, int s1, int s2, int s3)
{
	__shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	int parts = s1 / BLOCK_SIZE + 1;
	float ans = 0;

	for (int id = 0; id < parts; id++)
	{
		if (i < s1 && id * BLOCK_SIZE + threadIdx.x < s2)
		{
			a_shared[threadIdx.y][threadIdx.x] = a[i * s2 + id * BLOCK_SIZE + threadIdx.x];
		}
		else
		{
			a_shared[threadIdx.y][threadIdx.x] = 0;
		}

		if (j < s3 && id * BLOCK_SIZE + threadIdx.y < s2)
		{
			b_shared[threadIdx.y][threadIdx.x] = b[j + s3 * (id * BLOCK_SIZE + threadIdx.y)];
		}
		else
		{
			b_shared[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();
		for (int k = 0; k < BLOCK_SIZE; k++)
		{
			ans += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
		}
		__syncthreads();
	}

	if (i < s1 && j < s3)
	{
		c[i * s3 + j] = ans;
	}
}

void fillMat(float* a, int s)
{
	for (int i = 0; i < s; i++)
	{
		a[i] = rand() % 100;
		//a[i] = (0.8 * ((float)rand() / (float)RAND_MAX) + 0.1);
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
	int iterations = 1;
	int size = 1000;
	srand(time(NULL));
	float* a = (float*)malloc(size * size * sizeof(float));
	float* b = (float*)malloc(size * size * sizeof(float));
	float* c = (float*)malloc(size * size * sizeof(float));

	float* a_device;
	float* b_device;
	float* c_device;
	float* c_device_cublas;
	float* c_device_shared;
	cudaMalloc(&a_device, size * size * sizeof(float));
	cudaMalloc(&b_device, size * size * sizeof(float));
	cudaMalloc(&c_device, size * size * sizeof(float));
	cudaMalloc(&c_device_cublas, size * size * sizeof(float));
	cudaMalloc(&c_device_shared, size * size * sizeof(float));

	int grid_size = 1 + size / BLOCK_SIZE;

	dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 grid(grid_size, grid_size, 1);

	float sum_h;
	float sum_d;

	clock_t time_h = 0;
	float time_d_event = 0;
	float time_d_cublas = 0;
	float time_d_shared = 0;

	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0.0f;

	// warmup
	fillMat(a, size * size);
	fillMat(b, size * size);
	cudaMemcpy(a_device, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_device, b, size * size * sizeof(float), cudaMemcpyHostToDevice);
	matMulGpuShared<<<grid, block>>>(a_device, b_device, c_device, size, size, size);
	cudaDeviceSynchronize();
	 cudaError_t error = cudaGetLastError();
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, b_device, size, a_device, size,
			&beta, c_device_cublas, size);
	cudaDeviceSynchronize();
	// done

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("matrix size: %d iterations: %d\n", size, iterations);

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

		cudaEventRecord(start, 0);
		matMulGpu<<<grid, block>>>(a_device, b_device, c_device, size, size, size);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&time, start, stop);
		time_d_event += (time / 1000.0);

		cudaEventRecord(start, 0);
		matMulGpuShared<<<grid, block>>>(a_device, b_device, c_device_shared, size, size, size);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&time, start, stop);
		time_d_shared += (time / 1000.0);

		cudaEventRecord(start, 0);
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, b_device, size, a_device, size,
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
			printf("OK\n");
		}
		printf("host: %f\ndevice: %f\n", sum_h, sum_d);
		cudaMemcpy(c, c_device_shared, size * size * sizeof(float), cudaMemcpyDeviceToHost);
		sum_d = calcSumCpu(c, size * size);
		if (sum_h == sum_d)
		{
			printf("OK ");
		}
		printf("shared: %f\n", sum_d);

		cudaMemcpy(c, c_device_cublas, size * size * sizeof(float), cudaMemcpyDeviceToHost);
		sum_d = calcSumCpu(c, size * size);
		if (sum_h == sum_d)
		{
			printf("OK ");
		}
		printf("cublas: %f\n", sum_d);
	}

	float time_h_secs = (float)time_h / CLOCKS_PER_SEC;
	float profit_event = time_h_secs / time_d_event;
	float profit_shared = time_h_secs / time_d_shared;
	float profit_cublas = time_h_secs / time_d_cublas;
	printf("profit event: %f\nprofit shared: %f\nprofit cublas %f\ntime_h: %f\ntime_d_event: %f\ntime_d_shared: %f\ntime_d_cublas: %f\n\n",
		profit_event, profit_shared, profit_cublas, time_h_secs, time_d_event, time_d_shared, time_d_cublas);

	//fclose(stdout);
	status = cublasDestroy(handle);
	free(a);
	free(b);
	free(c);
	cudaFree(a_device);
	cudaFree(b_device);
	cudaFree(c_device);
	cudaFree(c_device_shared);
	cudaFree(c_device_cublas);

    return 0;
}

