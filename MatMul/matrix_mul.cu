#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <iostream>
#include <cublas_v2.h>

void matrix_mul_cpu(float* a, float* b, float* c, int a_height, int a_width, int b_width) // a * b = c; 
{
	for (int i = 0; i < a_height; i++)
	{
		for (int j = 0; j < b_width; j++)
		{
			float ans = 0;
			for (int k = 0; k < a_width; k++)
			{
				ans += a[i * a_width + k] * b[k * b_width + j];
			}
			c[i * b_width + j] = ans;
		}
	}
}


void fill_matrix(float* a, int size)
{
	for (int i = 0; i < size; i++)
	{
		a[i] = rand() % 100;
	}
}

void print_matrix(float* a, int height, int width)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			printf("%f ", a[i * width + j]);
		}
		printf("\n");
	}
	printf("\n");
}

float calculate_sum_cpu(float* a, int size)
{
	float ans = 0;
	for (int i = 0; i < size; i++)
	{
		ans += a[i];
	}
	return ans;
}

int main()
{
	//freopen("out.txt", "w", stdout);
	int iterations = 1;
	//int size = 1000;
	int a_height = 500;
	int a_width = 700;
	int b_height = 700;
	int b_width = 800;
	srand(time(NULL));
	float* a = (float*)malloc(a_height * a_width * sizeof(float));
	float* b = (float*)malloc(b_height * b_width * sizeof(float));
	float* c = (float*)malloc(a_height * b_width * sizeof(float));

	float* a_device;
	float* b_device;
	float* c_device;
	cudaMalloc(&a_device, a_height * a_width * sizeof(float));
	cudaMalloc(&b_device, b_height * b_width * sizeof(float));
	cudaMalloc(&c_device, a_height * b_width * sizeof(float));
	
	float sum_h;
	float sum_d;

	clock_t time_h = 0;
	float time_d = 0;
	
	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0.0f;

	// warmup
	fill_matrix(a, a_height * a_width);
	fill_matrix(b, b_height * b_width);
	cudaMemcpy(a_device, a, a_height * a_width * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_device, b, b_height * b_width * sizeof(float), cudaMemcpyHostToDevice);
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, b_width, a_height, a_width, &alpha, b_device, b_width, a_device,
			a_width, &beta, c_device, b_width);
	cudaDeviceSynchronize();
	// done

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < iterations; i++)
	{
		fill_matrix(a, a_height * a_width);
		fill_matrix(b, b_height * b_width);
		cudaMemcpy(a_device, a, b_height * b_width * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(b_device, b, b_height * b_width * sizeof(float), cudaMemcpyHostToDevice);

		clock_t t1 = clock();
		float time;
		matrix_mul_cpu(a, b, c, a_height, a_width, b_width);
		t1 = clock() - t1;
		time_h += t1;

		cudaEventRecord(start, 0);
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, b_width, a_height, a_width, &alpha, b_device, b_width, a_device,
			a_width, &beta, c_device, b_width);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		time_d += (time / 1000.0);

		sum_h = calculate_sum_cpu(c, a_height * b_width);
		print_matrix(c, a_height, b_width);
		cudaMemcpy(c, c_device, a_height * b_width * sizeof(float), cudaMemcpyDeviceToHost);
		sum_d = calculate_sum_cpu(c, a_height * b_width);
		print_matrix(c, a_height, b_width);
		if (sum_h == sum_d)
		{
			printf("OK! ");
		}
		printf("cpu sum: %f; cublas sum: %f\n", sum_d);
	}

	float time_h_secs = (float)time_h / CLOCKS_PER_SEC;
	float profit = time_h_secs / time_d;
	printf("profit: %f\n", profit);

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

