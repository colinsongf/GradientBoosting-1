#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <vector>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include "matrix_mul.cuh"

matrix_mul::matrix_mul(std::string a_file_name, std::string b_file_name, int a_size,
	int b_size, int features_size) : a_file_name(a_file_name), b_file_name(b_file_name),
	a_size(a_size),	b_size(b_size),	features_size(features_size) {}

void matrix_mul::calculate(std::string output_file_name, int n, int block_size)
{
	std::ifstream a_stream(a_file_name.c_str());
	std::ifstream b_stream(b_file_name.c_str());
	FILE* fout = fopen(output_file_name.c_str(), "w");
	std::vector<int> b_ids;
	std::vector<float> b(b_size * features_size);
	char const tab_delim = '\t';
	std::string line;
	thrust::device_vector<int> b_ids_device;
	for (int i = 0; i < b_size; i++) //read "b" matrix
	{
		if (i % 1000000 == 0)
		{
			printf("Reading B: %d of %d\n", i, b_size);
		}
		getline(b_stream, line);
		std::istringstream line_stream(line);
		std::string value;
		getline(line_stream, value, tab_delim);
		b_ids.push_back(atoi(value.c_str()));
		for (int j = 0; j < features_size; j++)
		{
			getline(line_stream, value, tab_delim);
			b[j * b_size + i] = atof(value.c_str());
		}
		b_ids_device.push_back(i);
	}
	thrust::device_vector<float> b_device(b);
	thrust::device_vector<float> sorted_values_d(b_size);
	thrust::device_vector<int> sorted_ids_d(b_size);
	
	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0.0f;
	
	int parts = a_size / block_size + 1;
	
	clock_t mul_time;
	clock_t total_time;
	clock_t io_time;
	clock_t sort_time;
	
	size_t temp_storage_bytes;
	void* d_temp_storage;

	for (int id = 0; id < parts; id++) //process "a" matrix by parts
	{
		cudaDeviceSynchronize();
		if (id % 100 == 0)
		{
			printf("Part %d of %d\n", id, parts);
			printf("mul time: %f sec\n", (float)mul_time / CLOCKS_PER_SEC);
			printf("i/o and data transfers time: %f sec\n", (float)io_time / CLOCKS_PER_SEC);
			printf("sort time: %f sec\n", (float)sort_time / CLOCKS_PER_SEC);
			printf("total time: %f sec\n\n", (float)total_time / CLOCKS_PER_SEC);
			mul_time = 0;
			total_time = 0;
			io_time = 0;
			sort_time = 0;
		}
		clock_t time = clock();
		int a_actual_size = (id == parts - 1) ? (a_size % block_size) : block_size;
		std::vector<int> a_ids;
		std::vector<float> a(a_actual_size * features_size);	
		for (int i = 0; (i < block_size) && (id * block_size + i < a_size); i++) //read part of "a" matrix
		{
			getline(a_stream, line);
			std::istringstream line_stream(line);
			std::string value;
			getline(line_stream, value, tab_delim);
			a_ids.push_back(atoi(value.c_str()));
			for (int j = 0; j < features_size; j++)
			{
				getline(line_stream, value, tab_delim);
				a[i * features_size + j] = (float)atof(value.c_str());
			}
		}
		thrust::device_vector<float> a_device(a);
		thrust::device_vector<float> c_device(a_actual_size * b_size);
		cudaDeviceSynchronize();
		clock_t time_temp = clock() - time;
		io_time += time_temp;
		
		time_temp = clock();
		// matrix multiplication
		status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, b_size, a_actual_size, features_size, &alpha,
			thrust::raw_pointer_cast(&b_device[0]), b_size, thrust::raw_pointer_cast(&a_device[0]),
			features_size, &beta, thrust::raw_pointer_cast(&c_device[0]), b_size);
		cudaDeviceSynchronize();
		time_temp = clock() - time_temp;
		mul_time += time_temp;
		
		if (id == 0 || id == (parts - 1)) //allocate memory for sort
		{
			temp_storage_bytes = 0;
			d_temp_storage = NULL;
			cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, thrust::raw_pointer_cast(&c_device[0]),
				thrust::raw_pointer_cast(&sorted_values_d[0]), thrust::raw_pointer_cast(&b_ids_device[0]),
				thrust::raw_pointer_cast(&sorted_ids_d[0]), b_size);
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
		}
		
		for (int i = 0; i < a_actual_size; i++) //select "n" best ids
		{
			cudaDeviceSynchronize();
			time_temp = clock();
			cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, thrust::raw_pointer_cast(&c_device[0]) + i * b_size,
				thrust::raw_pointer_cast(&sorted_values_d[0]), thrust::raw_pointer_cast(&b_ids_device[0]),
				thrust::raw_pointer_cast(&sorted_ids_d[0]), b_size);
			cudaDeviceSynchronize();
			time_temp = clock() - time_temp;
			sort_time += time_temp;
			
			time_temp = clock();
			thrust::host_vector<int> sorted_ids(sorted_ids_d.begin(), sorted_ids_d.begin() + n);
			fprintf(fout, "%d", a_ids[i]);
			for (int j = 0; j < n; j++)
			{
				fprintf(fout, "\t%d", b_ids[(int)sorted_ids[j]]);
			}
			fprintf(fout, "\n");
			cudaDeviceSynchronize();
			time_temp = clock() - time_temp;
			io_time += time_temp;
		}
		cudaDeviceSynchronize();
		time = clock() - time;
		total_time += time;
	}
	fclose(fout);
	cudaFree(d_temp_storage);
	status = cublasDestroy(handle);
}

void matrix_mul::print_c_matrix(float* c_device, int a_actual_size)
{
	float* c = (float*)malloc(a_actual_size * b_size * sizeof(float));
	cudaMemcpy(c, c_device, a_actual_size * b_size * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < a_actual_size; i++)
	{
		for (int j = 0; j < b_size; j++)
		{
			printf("%f ", c[i * b_size + j]);
		}
		printf("\n");
	}
	free(c);
}
